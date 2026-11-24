"""CPU checks for uninterrupted cosine decay and explicit schedule changes."""

import copy
import unittest

import torch
from torch import nn

from lr_schedule import build_scheduler, restore_training_state


class TinySupervisedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.main_head = nn.Linear(4, 2)
        self.auxiliary_heads = nn.ModuleDict({
            name: nn.Linear(4, 2) for name in ("rgb", "hsv", "lab")
        })

    def loss(self):
        inputs = torch.tensor([[0.1, 0.5, 0.9], [0.8, 0.3, 0.2]])
        features = self.encoder(inputs)
        main = self.main_head(features).square().mean()
        auxiliary = torch.stack([head(features).square().mean()
                                 for head in self.auxiliary_heads.values()]).mean()
        return main + 0.4 * auxiliary


class LearningRateScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)

    @staticmethod
    def config(kind="CosineAnnealingLR", epochs=50, learning_rate=6e-5):
        scheduler = {"type": kind, "eta_min": 1e-6}
        if kind == "CosineAnnealingWarmRestarts":
            scheduler.update(T_0=10, T_mult=2)
        else:
            scheduler["T_max"] = epochs
        return {"training": {"learning_rate": learning_rate, "epochs": epochs,
                             "scheduler": scheduler}}

    @staticmethod
    def optimizer(model, learning_rate):
        return torch.optim.AdamW([
            {"params": list(model.encoder.parameters()) + list(model.main_head.parameters())},
            {"params": model.auxiliary_heads.parameters()},
        ], lr=learning_rate)

    @staticmethod
    def step(model, optimizer, scheduler):
        used_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.zero_grad(set_to_none=True)
        model.loss().backward()
        optimizer.step()
        scheduler.step()
        return used_rates

    def checkpoint(self, config, steps):
        model = TinySupervisedModel()
        optimizer = self.optimizer(model, config["training"]["learning_rate"])
        scheduler = build_scheduler(optimizer, config)
        for _ in range(steps):
            self.step(model, optimizer, scheduler)
        saved = {
            "config": copy.deepcopy(config), "epoch": steps - 1, "best_ber": 21.708915,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        }
        return saved, model, optimizer, scheduler

    def test_cosine_decreases_monotonically_until_final_epoch(self):
        config = self.config(epochs=12)
        model = TinySupervisedModel()
        optimizer = self.optimizer(model, 6e-5)
        scheduler = build_scheduler(optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(12):
            self.step(model, optimizer, scheduler)
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(rates[0], 6e-5)
        self.assertTrue(all(next_rate <= rate for rate, next_rate in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[-1], 1e-6, places=15)

    def test_remaining_epoch_override_reaches_minimum_at_training_end(self):
        config = self.config(epochs=50)
        model = TinySupervisedModel()
        optimizer = self.optimizer(model, 1e-5)
        scheduler = build_scheduler(optimizer, config, epochs_remaining=7)
        self.assertEqual(scheduler.T_max, 7)
        for _ in range(7):
            self.step(model, optimizer, scheduler)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6, places=15)

    def test_resume_matches_uninterrupted_learning_rates_and_model_updates(self):
        for kind in ("CosineAnnealingLR", "CosineAnnealingWarmRestarts"):
            with self.subTest(scheduler=kind):
                config = self.config(kind=kind, epochs=12)
                saved, original, optimizer, scheduler = self.checkpoint(config, 3)
                resumed = TinySupervisedModel()
                resumed_optimizer = self.optimizer(resumed, config["training"]["learning_rate"])
                resumed_scheduler, start_epoch, best_ber = restore_training_state(
                    resumed, resumed_optimizer, saved, config,
                )
                self.assertEqual(start_epoch, 3)
                self.assertEqual(best_ber, saved["best_ber"])
                self.assertEqual([group["lr"] for group in resumed_optimizer.param_groups],
                                 [group["lr"] for group in optimizer.param_groups])
                for _ in range(start_epoch, 12):
                    expected = self.step(original, optimizer, scheduler)
                    actual = self.step(resumed, resumed_optimizer, resumed_scheduler)
                    self.assertEqual(actual, expected)
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)

    def test_reset_from_warm_restart_keeps_all_weights_and_adam_moments(self):
        previous = self.config(kind="CosineAnnealingWarmRestarts")
        saved, _, _, _ = self.checkpoint(previous, 10)
        self.assertEqual(saved["optimizer_state_dict"]["param_groups"][0]["lr"], 6e-5)
        config = self.config(learning_rate=1e-5)
        model = TinySupervisedModel()
        optimizer = self.optimizer(model, 1e-5)
        scheduler, start_epoch, best_ber = restore_training_state(
            model, optimizer, saved, config, reset_scheduler=True,
        )
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(start_epoch, 10)
        self.assertEqual(scheduler.T_max, 40)
        self.assertEqual(best_ber, saved["best_ber"])
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], 1e-5)
            self.assertEqual(group["initial_lr"], 1e-5)
        for name, value in saved["model_state_dict"].items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        for key, state in saved["optimizer_state_dict"]["state"].items():
            restored_state = optimizer.state_dict()["state"][key]
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(restored_state[name], state[name], rtol=0, atol=0)
        for parameter in model.auxiliary_heads.parameters():
            self.assertGreater(optimizer.state[parameter]["exp_avg"].abs().sum().item(), 0)
        self.assertEqual(self.step(model, optimizer, scheduler), [1e-5, 1e-5])
        self.assertTrue(all(group["lr"] <= 1e-5 for group in optimizer.param_groups))

    def test_schedule_and_learning_rate_changes_require_explicit_reset(self):
        old_config = self.config(kind="CosineAnnealingWarmRestarts")
        saved, _, _, _ = self.checkpoint(old_config, 3)
        changed_rate = copy.deepcopy(old_config)
        changed_rate["training"]["learning_rate"] = 1e-5
        changed_floor = copy.deepcopy(old_config)
        changed_floor["training"]["scheduler"]["eta_min"] = 2e-6
        alternatives = (self.config(), changed_rate, changed_floor)
        for config in alternatives:
            with self.subTest(config=config):
                model = TinySupervisedModel()
                optimizer = self.optimizer(model, config["training"]["learning_rate"])
                with self.assertRaises(ValueError):
                    restore_training_state(model, optimizer, saved, config)

    def test_reset_rejects_no_remaining_epochs(self):
        previous = self.config(kind="CosineAnnealingWarmRestarts")
        saved, _, _, _ = self.checkpoint(previous, 10)
        for epochs in (10, 9):
            with self.subTest(epochs=epochs):
                model = TinySupervisedModel()
                optimizer = self.optimizer(model, 1e-5)
                with self.assertRaises(ValueError):
                    restore_training_state(model, optimizer, saved,
                                           self.config(epochs=epochs, learning_rate=1e-5),
                                           reset_scheduler=True)


if __name__ == "__main__":
    unittest.main()

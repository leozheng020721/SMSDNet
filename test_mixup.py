"""CPU checks for paired MixUp images and supervised segmentation targets."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from torch import nn

from losses import FocalDiceLoss, SegmentationObjective
from mixup import BatchMixUp, mixup_objective


class BatchMixUpTest(unittest.TestCase):
    def setUp(self):
        self.images = torch.arange(36, dtype=torch.float32).reshape(3, 3, 2, 2)
        self.masks = torch.tensor([
            [[0, 1], [255, 0]], [[1, 0], [1, 255]], [[255, 1], [0, 1]],
        ], dtype=torch.long)

    def test_fixed_lambda_mixes_pixels_with_corresponding_unchanged_long_targets(self):
        original_images, original_masks = self.images.clone(), self.masks.clone()
        permutation = torch.tensor([2, 0, 1])
        with patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.25)), \
                patch("torch.randperm", return_value=permutation):
            images, target_a, target_b, lam = BatchMixUp(probability=1)(self.images, self.masks)
        self.assertEqual(lam, 0.25)
        torch.testing.assert_close(images, 0.25 * self.images + 0.75 * self.images[permutation])
        torch.testing.assert_close(target_a, self.masks)
        torch.testing.assert_close(target_b, self.masks[permutation])
        self.assertEqual(target_a.dtype, torch.long)
        self.assertEqual(target_b.dtype, torch.long)
        self.assertEqual(set(target_b.unique().tolist()), {0, 1, 255})
        torch.testing.assert_close(self.images, original_images)
        torch.testing.assert_close(self.masks, original_masks)

    def test_torch_seed_reproduces_sampling_and_pairing(self):
        results = []
        for _ in range(2):
            torch.manual_seed(71)
            results.append(BatchMixUp(alpha=0.2, probability=1)(self.images, self.masks))
        for first, second in zip(results[0][:3], results[1][:3]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(results[0][3], results[1][3])

    def test_disabled_probability_singleton_and_rejected_draw_skip_sampling(self):
        cases = ((0, self.images, self.masks), (1, self.images[:1], self.masks[:1]),
                 (0.5, self.images, self.masks))
        for probability, images, masks in cases:
            with self.subTest(probability=probability, batch=images.shape[0]):
                with patch("torch.rand", return_value=torch.tensor(0.9)), \
                        patch("torch.distributions.Beta.sample") as sample, \
                        patch("torch.randperm") as permute:
                    mixed, first, second, lam = BatchMixUp(probability=probability)(images, masks)
                self.assertIs(mixed, images)
                self.assertIs(first, masks)
                self.assertIs(second, masks)
                self.assertEqual(lam, 1.0)
                sample.assert_not_called()
                permute.assert_not_called()

    def test_invalid_configuration_and_targets_are_rejected(self):
        for alpha in (0, -0.2, float("inf"), float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                BatchMixUp(alpha=alpha)
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                BatchMixUp(probability=probability)
        with self.assertRaisesRegex(TypeError, "long"):
            BatchMixUp()(self.images, self.masks.float())
        with self.assertRaisesRegex(ValueError, "sizes"):
            BatchMixUp()(self.images, self.masks[:2])


class MixUpObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def outputs():
        return {
            "out": torch.randn(2, 2, 3, 4, requires_grad=True),
            "aux": {name: torch.randn(2, 2, 3, 4, requires_grad=True)
                    for name in ("rgb", "hsv", "lab")},
        }

    def test_all_components_are_weighted_and_main_and_auxiliary_gradients_flow(self):
        outputs = self.outputs()
        target_a = torch.randint(0, 2, (2, 3, 4))
        target_b = 1 - target_a
        target_a[:, 0, 0] = 255
        target_b[:, 0, 1] = 255
        objective = SegmentationObjective(nn.CrossEntropyLoss(ignore_index=255), 255, 0.4)
        lam = 0.3
        first, parts_a = objective(outputs, target_a, return_components=True)
        second, parts_b = objective(outputs, target_b, return_components=True)
        loss, parts = mixup_objective(objective, outputs, target_a, target_b, lam)
        torch.testing.assert_close(loss, lam * first + (1 - lam) * second)
        for name in ("main", "aux_rgb", "aux_hsv", "aux_lab", "aux", "total"):
            torch.testing.assert_close(parts[name], lam * parts_a[name] + (1 - lam) * parts_b[name])
        loss.backward()
        for logits in (outputs["out"], *outputs["aux"].values()):
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)
        scalar_only = mixup_objective(objective, outputs, target_a, target_b, lam, return_components=False)
        torch.testing.assert_close(scalar_only, loss)

    def test_focal_dice_ignores_padding_and_all_ignored_targets_have_zero_gradient(self):
        outputs = self.outputs()
        target = torch.full((2, 3, 4), 255, dtype=torch.long)
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        loss, parts = mixup_objective(objective, outputs, target, target.clone(), 0.3)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        loss.backward()
        for logits in (outputs["out"], *outputs["aux"].values()):
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_combo_mixup_excludes_each_sources_minmax_ignore_regions(self):
        outputs = self.outputs()
        target_a = torch.randint(0, 2, (2, 3, 4))
        target_b = 1 - target_a
        target_a[:, 0, :2] = 255
        target_b[:, 0, 0] = 255
        target_b[:, 1, 0] = 255
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        tensors = (outputs["out"], *outputs["aux"].values())
        loss_a = objective(outputs, target_a)
        loss_b = objective(outputs, target_b)
        gradients_a = torch.autograd.grad(loss_a, tensors, retain_graph=True)
        gradients_b = torch.autograd.grad(loss_b, tensors, retain_graph=True)
        loss = mixup_objective(objective, outputs, target_a, target_b, 0.3, return_components=False)
        loss.backward()
        common_ignore = ((target_a == 255) & (target_b == 255))[:, None]
        only_b_supervised = ((target_a == 255) & (target_b != 255))[:, None]
        for logits, grad_a, grad_b in zip(tensors, gradients_a, gradients_b):
            torch.testing.assert_close(logits.grad, 0.3 * grad_a + 0.7 * grad_b)
            self.assertEqual(logits.grad.masked_select(common_ignore).abs().sum().item(), 0)
            self.assertGreater(logits.grad.masked_select(only_b_supervised).abs().sum().item(), 0)

    def test_combo_mixup_with_one_all_ignored_source_keeps_the_other_target(self):
        outputs = self.outputs()
        target_a = torch.full((2, 3, 4), 255, dtype=torch.long)
        target_b = torch.randint(0, 2, (2, 3, 4))
        target_b[:, -1, -1] = 255
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        expected, expected_parts = objective(outputs, target_b, return_components=True)
        actual, parts = mixup_objective(objective, outputs, target_a, target_b, 0.3)
        torch.testing.assert_close(actual, 0.7 * expected)
        for name in parts:
            torch.testing.assert_close(parts[name], 0.7 * expected_parts[name])
        actual.backward()
        for logits in (outputs["out"], *outputs["aux"].values()):
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad[:, :, -1, -1].abs().sum().item(), 0)
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_endpoint_lambdas_only_evaluate_one_target(self):
        outputs = self.outputs()
        target_a = torch.zeros((2, 3, 4), dtype=torch.long)
        target_b = torch.ones_like(target_a)
        objective = SegmentationObjective(nn.CrossEntropyLoss(), 255, 0.4)
        for lam, target in ((1, target_a), (0, target_b)):
            with self.subTest(lam=lam), patch.object(objective, "forward", wraps=objective.forward) as forward:
                loss, _ = mixup_objective(objective, outputs, target_a, target_b, lam)
                forward.assert_called_once()
                self.assertIs(forward.call_args.args[1], target)
            torch.testing.assert_close(loss, objective(outputs, target))

    def test_invalid_lambda_is_rejected(self):
        for lam in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                mixup_objective(None, None, None, None, lam)

    def test_training_uses_one_forward_per_mixed_batch_and_validation_stays_unmixed(self):
        from torch.utils.data import DataLoader, TensorDataset
        from train import train_one_epoch, validate

        class TinyColorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = nn.ModuleDict({name: nn.Conv2d(3, 4, 1)
                                               for name in ("rgb", "hsv", "lab")})
                self.auxiliary_heads = nn.ModuleDict({name: nn.Conv2d(4, 2, 1)
                                                      for name in self.encoders})
                self.main_head = nn.Conv2d(12, 2, 1)
                self.main_calls = 0
                self.auxiliary_calls = 0

            def features(self, images):
                return {name: encoder(images) for name, encoder in self.encoders.items()}

            def forward(self, images):
                self.main_calls += 1
                return self.main_head(torch.cat(list(self.features(images).values()), dim=1))

            def forward_with_aux(self, images):
                self.auxiliary_calls += 1
                features = self.features(images)
                return {"out": self.main_head(torch.cat(list(features.values()), dim=1)),
                        "aux": {name: head(features[name])
                                for name, head in self.auxiliary_heads.items()}}

        torch.manual_seed(59)
        model = TinyColorModel()
        images = torch.rand(4, 3, 6, 7)
        masks = torch.randint(0, 2, (4, 6, 7))
        masks[:, -1, -1] = 255
        loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
        objective = SegmentationObjective(FocalDiceLoss({"focal_weight": 2.0}, 255), 255, 0.4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        modules = {**dict(model.encoders.items()), "main_head": model.main_head,
                   **{"aux_" + name: head for name, head in model.auxiliary_heads.items()}}
        before = {name: next(module.parameters()).detach().clone() for name, module in modules.items()}
        config = {"training": {"mixup": {"enabled": True, "alpha": 0.2, "probability": 1}},
                  "dataset": {"spatial_mode": "resize"}}
        with patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.25)), \
                patch("torch.randperm", return_value=torch.tensor([1, 0])), \
                contextlib.redirect_stderr(io.StringIO()):
            stats = train_one_epoch(model, loader, objective, optimizer, None, "cpu", config=config)
        self.assertEqual(stats["mixup_fraction"], 1.0)
        self.assertEqual(stats["mixup_lambda"], 0.25)
        self.assertEqual(model.auxiliary_calls, len(loader))
        self.assertEqual(model.main_calls, 0)
        self.assertAlmostEqual(stats["total"], stats["main"] + 0.4 * stats["aux"], places=6)
        for name, module in modules.items():
            self.assertFalse(torch.equal(next(module.parameters()).detach(), before[name]), name)
        with patch("train.BatchMixUp") as create_mixer, \
                patch("train.mixup_objective") as mixed_objective, \
                contextlib.redirect_stderr(io.StringIO()):
            val_loss, metrics = validate(model, loader, objective, "cpu", 2, config=config)
        create_mixer.assert_not_called()
        mixed_objective.assert_not_called()
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["ber"])))
        self.assertEqual(model.main_calls, len(loader))
        self.assertEqual(model.auxiliary_calls, len(loader))


if __name__ == "__main__":
    unittest.main()

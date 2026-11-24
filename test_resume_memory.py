"""Small CPU resume checks for checkpoint lifetime and writable output files."""

import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from lr_schedule import build_scheduler
from train import main


class _WeakCheckpoint(dict):
    pass


class _RetentionProbe:
    pass


class ResumeMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def config(experiment, epochs):
        return {
            "random_seed": 42, "device": "cpu", "num_workers": 0,
            "pin_memory": False, "cudnn_benchmark": False,
            "experiment_name": experiment,
            "model": {"variant": "baseline", "encoder": "mit_b0", "ignore_index": 255},
            "dataset": {
                "type": "seo", "root_dir": "unused", "val_ratio": 0.2,
                "spatial_mode": "resize", "input_size": 4,
                "in_channels": 3, "num_classes": 2,
                "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
            },
            "training": {
                "batch_size": 2, "epochs": epochs, "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "scheduler": {"type": "CosineAnnealingLR", "T_max": epochs, "eta_min": 1e-5},
            },
            "loss": {"type": "ce"},
        }

    def run_resume(self, reset_scheduler, saved_supervision=None, requested_supervision=None,
                   supervision_reports=None):
        epochs = 3 if reset_scheduler else 2
        saved_config = self.config("source", epochs)
        if saved_supervision is not None:
            saved_config['dataset']['supervision'] = copy.deepcopy(saved_supervision)
        requested = copy.deepcopy(saved_config)
        if requested_supervision is not None:
            requested['dataset']['supervision'] = copy.deepcopy(requested_supervision)
        if reset_scheduler:
            requested["experiment_name"] = "reset"
            requested["training"]["learning_rate"] = 1e-4
        torch.manual_seed(15)
        source = nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(
            source.parameters(), lr=saved_config["training"]["learning_rate"], weight_decay=0.01
        )
        scheduler = build_scheduler(source_optimizer, saved_config)
        inputs = torch.tensor([[0.1, 0.3, 0.8], [0.9, 0.2, 0.4]])
        labels = torch.tensor([0, 1])
        F.cross_entropy(source(inputs), labels).backward()
        source_optimizer.step()
        scheduler.step()
        expected_model = copy.deepcopy(source.state_dict())
        expected_optimizer = copy.deepcopy(source_optimizer.state_dict())
        expected_rate = (requested["training"]["learning_rate"] if reset_scheduler
                         else source_optimizer.param_groups[0]["lr"])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        val_loader = loader
        if supervision_reports is not None:
            val_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
            loader.dataset.supervision_filter_report = supervision_reports['train']
            val_loader.dataset.supervision_filter_report = supervision_reports['val']
        checkpoint_refs, probe_refs, training_rates = [], [], []
        real_load = torch.load

        def tracked_load(*args, **kwargs):
            self.assertNotIn("mmap", kwargs)
            loaded = _WeakCheckpoint(real_load(*args, **kwargs))
            # Also detect a retained shallow starting_best copy on reset.
            loaded["_retention_probe"] = _RetentionProbe()
            checkpoint_refs.append(weakref.ref(loaded))
            probe_refs.append(weakref.ref(loaded["_retention_probe"]))
            return loaded

        def train_step(model, batches, criterion, optimizer, scaler, device, config):
            self.assertEqual(len(checkpoint_refs), 1)
            self.assertIsNone(checkpoint_refs[0](), "Loaded checkpoint container is still retained")
            self.assertIsNone(probe_refs[0](), "A shallow checkpoint preservation copy is still retained")
            self.assertFalse(scaler.is_enabled())
            if not training_rates:
                for name, value in expected_model.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                state = optimizer.state_dict()["state"]
                for key, values in expected_optimizer["state"].items():
                    for name, value in values.items():
                        torch.testing.assert_close(state[key][name], value, rtol=0, atol=0)
                self.assertEqual(optimizer.param_groups[0]["lr"], expected_rate)
                if reset_scheduler:
                    initial_best = real_load(best_path, map_location="cpu", weights_only=False)
                    self.assertEqual(initial_best["epoch"], 0)
                    self.assertEqual(initial_best["best_ber"], 12.5)
                    self.assertEqual(initial_best["config"]["experiment_name"], "reset")
                    self.assertEqual(initial_best["scheduler_state_dict"]["T_max"], epochs - 1)
            training_rates.append(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
            value = loss.detach().item()
            return {"main": value, "aux": 0.0, "total": value}

        metrics = {"ber": 20.0, "iou": 0.5, "f1": 0.6, "kappa": 0.4,
                   "shadow_miss": 20.0, "background_false_positive": 20.0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoints").mkdir()
            checkpoint_path = root / "checkpoints" / "source_last.pth"
            torch.save({
                "config": saved_config, "epoch": 0, "ber": 12.5, "best_ber": 12.5,
                "model_state_dict": source.state_dict(),
                "optimizer_state_dict": source_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": {},
            }, checkpoint_path)
            original_bytes = checkpoint_path.read_bytes() if reset_scheduler else None
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(requested), encoding="utf-8")
            best_path = root / "checkpoints" / "reset_best.pth"
            arguments = ["train.py", "--config", str(config_path), "--resume", str(checkpoint_path)]
            if reset_scheduler:
                arguments.append("--reset-scheduler")
            captured = io.StringIO()
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", arguments), \
                        patch("train.torch.load", side_effect=tracked_load), \
                        patch("train.build_model", side_effect=lambda _: nn.Linear(3, 2)), \
                        patch("train.get_seo_loaders", return_value=(loader, val_loader)), \
                        patch("train.torch.cuda.is_available", return_value=False), \
                        patch("train.train_one_epoch", side_effect=train_step), \
                        patch("train.validate", return_value=(1.0, metrics)), \
                        contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()):
                    main()
            finally:
                os.chdir(previous_directory)
            self.assertIn("Resuming from epoch 2", captured.getvalue())
            self.assertEqual(len(training_rates), epochs - 1)
            latest_path = root / "checkpoints" / f"{requested['experiment_name']}_last.pth"
            latest = real_load(latest_path, map_location="cpu", weights_only=False)
            self.assertEqual(latest["epoch"], epochs - 1)
            self.assertEqual(latest["best_ber"], 12.5)
            self.assertEqual(latest["config"], requested)
            self.assertAlmostEqual(latest["optimizer_state_dict"]["param_groups"][0]["lr"], 1e-5)
            for state in latest["optimizer_state_dict"]["state"].values():
                self.assertEqual(state["step"].item(), epochs)
            if reset_scheduler:
                self.assertEqual(checkpoint_path.read_bytes(), original_bytes)
                self.assertEqual(latest["scheduler_state_dict"]["T_max"], epochs - 1)
            else:
                self.assertEqual(latest_path, checkpoint_path)
                self.assertEqual(latest["scheduler_state_dict"]["last_epoch"], epochs)
            return captured.getvalue()

    def assert_protocol_resume_rejected(self, saved_supervision, requested_supervision,
                                       reset_scheduler, include_saved_config=True):
        saved = self.config('source', 2)
        if saved_supervision is not None:
            saved['dataset']['supervision'] = copy.deepcopy(saved_supervision)
        requested = self.config('source_minmax', 2)
        requested['dataset']['supervision'] = copy.deepcopy(requested_supervision)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path, config_path = root / 'source.pth', root / 'config.yaml'
            checkpoint = {'epoch': 0, 'ber': 2.5, 'best_ber': 2.5, 'model_variant': 'baseline'}
            if include_saved_config:
                checkpoint['config'] = saved
            torch.save(checkpoint, checkpoint_path)
            protected = checkpoint_path.read_bytes()
            config_path.write_text(yaml.safe_dump(requested), encoding='utf-8')
            arguments = ['train.py', '--config', str(config_path), '--resume', str(checkpoint_path)]
            if reset_scheduler:
                arguments.append('--reset-scheduler')
            with patch.object(sys, 'argv', arguments), \
                    patch('train.get_seo_loaders') as loaders, \
                    patch('train.build_model') as model, \
                    patch('train.restore_training_state') as restore, \
                    patch('train.torch.save') as save, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'dataset.supervision.*--init-from'):
                    main()
            loaders.assert_not_called()
            model.assert_not_called()
            restore.assert_not_called()
            save.assert_not_called()
            self.assertEqual(checkpoint_path.read_bytes(), protected)

    def test_cross_protocol_resume_and_scheduler_reset_reject_old_best_before_loading_data(self):
        protocols = ((None, {'mode': 'minmax_agreement'}),
                     ({'mode': 'minmax_agreement'}, {'mode': 'max'}))
        for saved, requested in protocols:
            for reset in (False, True):
                with self.subTest(saved=saved, requested=requested, reset=reset):
                    self.assert_protocol_resume_rejected(saved, requested, reset)

    def test_uncertainty_semantics_cannot_change_even_with_scheduler_reset(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                self.assert_protocol_resume_rejected(
                    {'mode': 'minmax_agreement', 'ignore_uncertainty': True},
                    {'mode': 'minmax_agreement', 'ignore_uncertainty': False}, reset)

    def test_checkpoint_without_config_is_legacy_max_and_cannot_resume_as_minmax(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                self.assert_protocol_resume_rejected(None, {'mode': 'minmax_agreement'}, reset,
                                                    include_saved_config=False)

    def test_missing_policy_changes_are_rejected_for_resume_and_scheduler_reset(self):
        protocols = (({'mode': 'minmax_agreement'},
                      {'mode': 'minmax_agreement', 'missing_policy': 'skip'}),
                     ({'mode': 'minmax_agreement', 'missing_policy': 'skip'},
                      {'mode': 'minmax_agreement', 'missing_policy': 'error'}))
        for saved, requested in protocols:
            for reset in (False, True):
                with self.subTest(saved=saved, requested=requested, reset=reset):
                    self.assert_protocol_resume_rejected(saved, requested, reset)

    def test_matching_semantics_allow_relocated_roots_and_print_active_protocol(self):
        output = self.run_resume(False,
            saved_supervision={'mode': 'minmax_agreement', 'min_root': 'old_min', 'max_root': 'old_max'},
            requested_supervision={'mode': 'minmax_agreement', 'ignore_uncertainty': True,
                                   'missing_policy': 'error',
                                   'min_root': 'new_min', 'max_root': 'new_max'})
        self.assertIn('SEO supervision: mode=minmax_agreement, ignore_uncertainty=True', output)
        self.assertIn('new_min', output)
        self.assertIn('new_max', output)
        self.assertIn('missing_policy=error', output)

    def test_startup_prints_both_supervision_filter_reports(self):
        reports = {
            'train': {'split': 'train', 'before': 3, 'after': 2, 'missing': 1,
                      'missing_examples': ['JAX_0/view'], 'missing_policy': 'skip'},
            'val': {'split': 'val', 'before': 2, 'after': 2, 'missing': 0,
                    'missing_examples': [], 'missing_policy': 'skip'},
        }
        options = {'mode': 'minmax_agreement', 'missing_policy': 'skip'}
        output = self.run_resume(False, saved_supervision=options, requested_supervision=options,
                                 supervision_reports=reports)
        self.assertIn('missing_policy=skip', output)
        for split, report in reports.items():
            self.assertIn(f'SEO supervision filter [{split}]: {report}', output)

    def test_regular_resume_releases_checkpoint_before_training_and_overwrites_last(self):
        self.run_resume(reset_scheduler=False)

    def test_reset_releases_preserved_copy_and_keeps_optimizer_schedule_and_source(self):
        self.run_resume(reset_scheduler=True)


if __name__ == "__main__":
    unittest.main()

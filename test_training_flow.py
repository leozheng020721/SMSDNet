"""CPU integration checks for supervised native crops and full-image validation."""

import contextlib
import copy
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
import yaml

from dataset import get_seo_loaders
from losses import build_criterion
from lr_schedule import build_scheduler
from metrics import fast_hist, get_metrics
from model_multicolor import MultiColorSegFormer
from patch_inference import sliding_window_logits
from train import main, train_one_epoch, validate


class NativeCropTrainingFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def config(root):
        return {
            "random_seed": 42, "device": "cpu", "num_workers": 0, "pin_memory": False,
            "model": {
                "variant": "multicolor", "encoder": "mit_b0", "encoder_weights": None,
                "ignore_index": 255, "multicolor": {
                    "aux_channels": [4, 8, 12, 16], "gate_channels": 4,
                    "auxiliary_supervision": True, "aux_decoder_channels": 8,
                },
            },
            "dataset": {
                "type": "seo", "root_dir": str(root), "val_ratio": 1 / 3,
                "spatial_mode": "crop", "input_size": 64,
                "tile_stride": 48, "tile_batch_size": 2,
                "in_channels": 3, "num_classes": 2,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                },
            },
            "training": {"batch_size": 2},
            "loss": {"type": "combo", "focal_weight": 2.0, "dice_weight": 1.0,
                     "auxiliary_weight": 0.4},
        }

    @staticmethod
    def write_dataset(root):
        for index, (height, width) in enumerate(((83, 109), (73, 91), (79, 101))):
            group = f"JAX_{index}"
            image_dir = root / "pansharpened_he" / group
            mask_dir = root / "shadows_max" / "shadow_max_compressed" / "shadow_masks" / group
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            rows, columns = np.indices((height, width))
            image = np.stack(((rows * 3) % 255, (columns * 5) % 255,
                              (rows + columns) % 255), axis=-1).astype(np.uint8)
            mask = np.where((rows // 5 + columns // 7) % 2 == 0, 0, 255).astype(np.uint8)
            if not cv2.imwrite(str(image_dir / "view_rgb.png"), image):
                raise RuntimeError("Unable to write test image")
            if not cv2.imwrite(str(mask_dir / "view_shadow.png"), mask):
                raise RuntimeError("Unable to write test mask")

    @staticmethod
    def make_model():
        return MultiColorSegFormer(
            encoder_name="mit_b0", encoder_weights=None, classes=2,
            decoder_segmentation_channels=32, aux_channels=(4, 8, 12, 16),
            gate_channels=4, auxiliary_supervision=True, aux_decoder_channels=8,
        )

    def test_crop_training_updates_all_branches_and_stitched_validation_matches_histogram(self):
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_dataset(root)
            config = self.config(root)
            train_loader, val_loader = get_seo_loaders(config)
            self.assertEqual(len(train_loader), 1)
            self.assertEqual(val_loader.batch_size, 1)
            images, masks = next(iter(train_loader))
            self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
            self.assertEqual(tuple(masks.shape), (2, 64, 64))
            model = self.make_model()
            criterion = build_criterion(config, "cpu")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            modules = {
                "rgb": model.encoder, "hsv": model.hsv_encoder, "lab": model.lab_encoder,
                **{"head_" + name: head for name, head in model.auxiliary_heads.items()},
            }
            parameters = {name: next(module.parameters()) for name, module in modules.items()}
            before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            with contextlib.redirect_stderr(io.StringIO()):
                losses = train_one_epoch(model, train_loader, criterion, optimizer, None, "cpu")
                val_loss, actual = validate(model, val_loader, criterion, "cpu", 2, config=config)
            self.assertTrue(all(np.isfinite(value) for value in losses.values()))
            for name in ("main", "aux_rgb", "aux_hsv", "aux_lab", "aux", "total"):
                self.assertIn(name, losses)
            self.assertAlmostEqual(losses["total"], losses["main"] + 0.4 * losses["aux"], places=6)
            for name, parameter in parameters.items():
                self.assertFalse(torch.equal(parameter.detach(), before[name]), name)
            self.assertTrue(np.isfinite(val_loss))

            histogram = np.zeros((2, 2), dtype=np.int64)
            total_pixels = 0
            for full_image, full_mask in val_loader:
                self.assertGreater(full_image.shape[-2], 64)
                self.assertGreater(full_image.shape[-1], 64)
                logits = sliding_window_logits(model, full_image, "cpu", 64, 48, 2)
                self.assertEqual(tuple(logits.shape[-2:]), tuple(full_mask.shape[-2:]))
                histogram += fast_hist(logits.argmax(dim=1), full_mask, 2)
                total_pixels += full_mask.numel()
            self.assertEqual(int(histogram.sum()), total_pixels)
            for name, value in get_metrics(histogram).items():
                self.assertAlmostEqual(actual[name], value, places=10)
            tp, fn = histogram[1, 1], histogram[1, 0]
            tn, fp = histogram[0, 0], histogram[0, 1]
            self.assertAlmostEqual(actual["shadow_miss"], 100.0 * fn / (tp + fn))
            self.assertAlmostEqual(actual["background_false_positive"], 100.0 * fp / (tn + fp))

    def test_resume_rejects_spatial_mode_change_before_loading_data_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root / "not_created")
            saved = copy.deepcopy(config)
            saved["dataset"]["spatial_mode"] = "resize"
            checkpoint_path = root / "old.pth"
            config_path = root / "config.yaml"
            torch.save({"config": saved}, checkpoint_path)
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            arguments = ["train.py", "--config", str(config_path), "--resume", str(checkpoint_path)]
            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, r"spatial_mode.*--init-from"):
                    main()

    def test_resume_rejects_validation_setting_changes_before_loading_data_or_model(self):
        changes = ((None, "random_seed", 17), ("dataset", "val_ratio", 0.5),
                   ("dataset", "tile_stride", 32))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root / "not_created")
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            checkpoint_path = root / "old.pth"
            arguments = ["train.py", "--config", str(config_path), "--resume", str(checkpoint_path)]
            for section, key, previous_value in changes:
                with self.subTest(changed=key):
                    saved = copy.deepcopy(config)
                    settings = saved if section is None else saved[section]
                    settings[key] = previous_value
                    torch.save({"config": saved}, checkpoint_path)
                    with patch.object(sys, "argv", arguments), \
                            patch("train.get_seo_loaders") as load_data, \
                            patch("train.build_model") as make_model:
                        with self.assertRaisesRegex(ValueError, key):
                            main()
                        load_data.assert_not_called()
                        make_model.assert_not_called()

    def test_reset_scheduler_requires_resume(self):
        with patch.object(sys, "argv", ["train.py", "--reset-scheduler"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    def test_auxiliary_head_transfer_requires_init_from(self):
        with patch.object(sys, "argv", ["train.py", "--init-auxiliary-heads"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    def test_resume_rejects_new_attention_before_loading_data_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = self.config(root / "not_created")
            requested = copy.deepcopy(saved)
            requested["model"]["multicolor"]["cross_scale_attention"] = {
                "enabled": True, "attention_channels": 64,
            }
            checkpoint_path = root / "old_aux.pth"
            config_path = root / "cca.yaml"
            torch.save({"config": saved}, checkpoint_path)
            config_path.write_text(yaml.safe_dump(requested), encoding="utf-8")
            arguments = ["train.py", "--config", str(config_path), "--resume", str(checkpoint_path)]
            with patch.object(sys, "argv", arguments), \
                    patch("train.get_seo_loaders") as load_data, \
                    patch("train.build_model") as make_model:
                with self.assertRaisesRegex(ValueError, r"model.multicolor"):
                    main()
            load_data.assert_not_called()
            make_model.assert_not_called()

    def test_reset_scheduler_saves_new_run_without_overwriting_original_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_dataset(root / "data")
            config = self.config(root / "data")
            config.update(experiment_name="new_cosine", cudnn_benchmark=False)
            config["training"].update(
                epochs=2, learning_rate=1e-5, weight_decay=0.01,
                scheduler={"type": "CosineAnnealingLR", "T_max": 2, "eta_min": 1e-6},
            )
            saved_config = copy.deepcopy(config)
            saved_config["experiment_name"] = "old_warm"
            saved_config["training"].update(
                learning_rate=6e-5,
                scheduler={"type": "CosineAnnealingWarmRestarts", "T_0": 1, "T_mult": 2},
            )
            source = self.make_model()
            optimizer = torch.optim.AdamW(source.parameters(), lr=6e-5, weight_decay=0.01)
            scheduler = build_scheduler(optimizer, saved_config)
            train_loader, _ = get_seo_loaders(saved_config)
            with contextlib.redirect_stderr(io.StringIO()):
                train_one_epoch(source, train_loader, build_criterion(saved_config, "cpu"),
                                optimizer, None, "cpu")
            scheduler.step()
            checkpoint_path = root / "old.pth"
            torch.save({
                "config": saved_config, "epoch": 0, "best_ber": 21.708915,
                "model_state_dict": source.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_path)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            (root / "checkpoints").mkdir()
            original_history = root / "checkpoints" / "old_warm_history.csv"
            original_text = "epoch,ber\n1,21.708915\n"
            original_history.write_text(original_text, encoding="utf-8")
            arguments = ["train.py", "--config", str(config_path), "--resume", str(checkpoint_path),
                         "--reset-scheduler"]
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", arguments), \
                        patch("train.build_model", side_effect=lambda _: self.make_model()), \
                        patch("train.torch.cuda.is_available", return_value=False), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    main()
            finally:
                os.chdir(previous_directory)
            self.assertEqual(original_history.read_text(encoding="utf-8"), original_text)
            with (root / "checkpoints" / "new_cosine_history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 1)
            self.assertEqual(int(history[0]["epoch"]), 2)
            self.assertEqual(float(history[0]["learning_rate"]), 1e-5)
            latest = torch.load(root / "checkpoints" / "new_cosine_last.pth",
                                map_location="cpu", weights_only=False)
            self.assertEqual(latest["epoch"], 1)
            self.assertEqual(latest["config"]["experiment_name"], "new_cosine")
            self.assertEqual(latest["scheduler_state_dict"]["T_max"], 1)
            self.assertAlmostEqual(latest["optimizer_state_dict"]["param_groups"][0]["lr"], 1e-6)
            self.assertTrue(any(name.startswith("auxiliary_heads.") for name in latest["model_state_dict"]))
            for state in latest["optimizer_state_dict"]["state"].values():
                self.assertEqual(state["step"].item(), 2)


if __name__ == "__main__":
    unittest.main()

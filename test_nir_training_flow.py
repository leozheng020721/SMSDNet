"""End-to-end NIR training and native-resolution inference checks."""

import contextlib
import csv
import io
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
from PIL import Image
import torch

from dataset import get_seo_loaders
from inference import build_inference_dataset, inference_single_image, predict_logits
from losses import SegmentationObjective, build_criterion
from model_factory import build_model
from patch_inference import sliding_window_logits
from train import train_one_epoch, validate


class NIRTrainingFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def fixture(root, minmax=False):
        nir_root = root / "nir_aligned_v1"
        records = []
        for index in range(4):
            group = "JAX_" + str(index)
            rgb_dir = root / "pansharpened_he" / group
            mask_dir = root / "shadows_max/shadow_max_compressed/shadow_masks" / group
            nir_dir = nir_root / group
            for directory in (rgb_dir, mask_dir, nir_dir):
                directory.mkdir(parents=True)
            rows, columns = np.indices((73 + index, 83 + index))
            rgb = np.stack(((rows * 3) % 255, (columns * 5) % 255,
                            (rows + columns * 2) % 255), axis=-1).astype(np.uint8)
            label = np.where((rows // 5 + columns // 7) % 2, 255, 0).astype(np.uint8)
            nir = (400 + rows * 4 + columns * 2).astype(np.float32)
            valid = (rows > 3) & (columns > 5)
            nir[~valid] = np.nan
            nir_path, valid_path = nir_dir / "view_nir.tif", nir_dir / "view_valid.png"
            Image.fromarray(nir).save(nir_path)
            for path, array in ((rgb_dir / "view_rgb.png", rgb),
                                (mask_dir / "view_shadow.png", label),
                                (valid_path, valid.astype(np.uint8) * 255)):
                if not cv2.imwrite(str(path), array):
                    raise OSError(str(path))
            if minmax:
                minimum = root / 'shadows_min/shadows/shadow_masks' / group
                uncertainty = root / 'shadows_min/shadows/uncertainty_masks' / group
                minimum.mkdir(parents=True)
                uncertainty.mkdir(parents=True)
                min_label = label.copy()
                min_label[10:20, 10:20] = 255 - min_label[10:20, 10:20]
                uncertain = np.zeros_like(label)
                uncertain[25:30, :] = 255
                cv2.imwrite(str(minimum / 'view_shadow.png'), min_label)
                cv2.imwrite(str(uncertainty / 'view_uncertainty_mask.png'), uncertain)
            records.append(dict(key=group + "/view", status="original_rgb_ready",
                                reference_grid="RGB", nir=str(nir_path), valid=str(valid_path)))
        with (nir_root / "registration_with_candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        config = {
            "random_seed": 42, "device": "cpu", "num_workers": 0, "pin_memory": False,
            "model": {"variant": "multicolor", "encoder": "mit_b0", "encoder_weights": None,
                      "ignore_index": 255, "multicolor": {
                          "aux_channels": [4, 8, 12, 16], "gate_channels": 4,
                          "auxiliary_supervision": True, "aux_decoder_channels": 8,
                          "nir": {"enabled": True},
                          "cross_scale_attention": {"enabled": True, "attention_channels": 8}}},
            "dataset": {"type": "seo", "root_dir": str(root), "val_ratio": 0.25,
                        "spatial_mode": "crop", "input_size": 64, "tile_stride": 48,
                        "tile_batch_size": 2, "in_channels": 3, "num_classes": 2,
                        "normalization": {"mean": [0.485, 0.456, 0.406],
                                          "std": [0.229, 0.224, 0.225]},
                        "nir": {"enabled": True, "root_dir": str(nir_root),
                                "normalization": {"percentiles": [1, 99]}}},
            "training": {"batch_size": 2, "mixup": {"enabled": True, "probability": 1.0}},
            "loss": {"type": "combo", "focal_weight": 2.0, "dice_weight": 1.0,
                     "auxiliary_weight": 0.4},
        }
        if minmax:
            config['dataset']['supervision'] = {'mode': 'minmax_agreement'}
        return config

    def test_real_loader_mixup_backward_and_native_validation(self):
        self.check_loader_mixup_backward_and_native_validation(minmax=False)

    def test_minmax_loader_mixup_backward_and_filtered_native_validation(self):
        self.check_loader_mixup_backward_and_native_validation(minmax=True)

    def check_loader_mixup_backward_and_native_validation(self, minmax):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory), minmax=minmax)
            train_loader, val_loader = get_seo_loaders(config)
            model = build_model(config)
            criterion = build_criterion(config, "cpu")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            modules = {"encoder": model.nir_encoder, "fusion": model.nir_fusions[0],
                       "head": model.auxiliary_heads["nir"]}
            parameters = {name: next(module.parameters()) for name, module in modules.items()}
            before = {name: value.detach().clone() for name, value in parameters.items()}
            with contextlib.redirect_stderr(io.StringIO()):
                statistics = train_one_epoch(model, train_loader, criterion, optimizer, None,
                                             "cpu", config)
                val_loss, metrics = validate(model, val_loader, criterion, "cpu", 2, config)
            self.assertTrue(all(np.isfinite(value) for value in statistics.values()))
            self.assertAlmostEqual(statistics["mixup_fraction"], 1.0)
            expected_aux = sum(statistics["aux_" + name] for name in ("rgb", "hsv", "lab", "nir")) / 4
            self.assertAlmostEqual(statistics["aux"], expected_aux, places=6)
            for name, value in parameters.items():
                self.assertFalse(torch.equal(value.detach(), before[name]), name)
            self.assertTrue(np.isfinite(val_loss))
            self.assertTrue(np.isfinite(metrics["ber"]))

            batch = next(iter(val_loader))
            if minmax:
                self.assertTrue((batch['mask'] == 255).any())
                self.assertTrue((batch['mask'] != 255).any())
            with torch.no_grad():
                logits = predict_logits(model, batch["image"], config, "cpu",
                                        nir=batch["nir"], nir_valid=batch["nir_valid"])
            self.assertEqual(tuple(logits.shape[-2:]), tuple(batch["mask"].shape[-2:]))
            expected_loss = criterion(logits, batch["mask"]).item()
            self.assertAlmostEqual(val_loss, expected_loss, places=6)

            # Single-image inference must find and transform the same NIR pair.
            single = inference_single_image(model, val_loader.dataset.img_paths[0], config, "cpu")
            np.testing.assert_array_equal(single, logits.argmax(1)[0].numpy())
            # Exporting the training split uses validation geometry as before.
            inference_train = build_inference_dataset(config, "train", count=1)
            self.assertGreater(inference_train[0]["image"].shape[-1], 64)

    def test_nir_tiles_preserve_coordinates_validity_and_small_image_padding(self):
        class PixelModel(torch.nn.Module):
            def forward(self, images, nir=None, nir_valid=None):
                value = torch.where(nir_valid, nir, images[:, :1])
                return torch.cat((value, -value), dim=1)

        for height, width in ((3, 5), (11, 13)):
            images = torch.arange(3 * height * width).reshape(1, 3, height, width).float()
            nir = images[:, :1] / 7
            valid = (nir.long() % 2) == 0
            expected = PixelModel()(images, nir, valid)
            actual = sliding_window_logits(PixelModel(), images, "cpu", 8, 5, 3,
                                           nir=nir, nir_valid=valid)
            torch.testing.assert_close(actual, expected)

    def test_fully_ignored_half_precision_batch_has_finite_zero_loss_and_gradients(self):
        tensors = [torch.full((1, 2, 64, 64), 65000, dtype=torch.float16, requires_grad=True)
                   for _ in range(5)]
        outputs = {"out": tensors[0], "aux": dict(zip(("rgb", "hsv", "lab", "nir"), tensors[1:]))}
        target = torch.full((1, 64, 64), 255, dtype=torch.long)
        valid = torch.zeros((1, 1, 64, 64), dtype=torch.bool)
        objective = SegmentationObjective(torch.nn.CrossEntropyLoss(ignore_index=255), 255, 0.4)
        loss = objective(outputs, target, auxiliary_valid={"nir": valid})
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for value in tensors:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertEqual(torch.count_nonzero(value.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()

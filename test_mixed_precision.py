"""Small end-to-end checks for CUDA FP16 and its CPU fallback."""

import contextlib
import io
import os
import unittest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from torch import nn

from losses import SegmentationObjective
from train import train_one_epoch, use_fp16, validate


class TinyNIRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(4, 8, 1)
        self.main = nn.Conv2d(8, 2, 1)
        self.heads = nn.ModuleDict({name: nn.Conv2d(8, 2, 1)
                                    for name in ("rgb", "hsv", "lab", "nir")})
        self.output_dtypes = []

    def features(self, images, nir, nir_valid):
        valid_nir = torch.where(nir_valid, nir, torch.zeros_like(nir))
        return torch.tanh(self.feature(torch.cat((images, valid_nir), dim=1)))

    def forward(self, images, nir=None, nir_valid=None):
        logits = self.main(self.features(images, nir, nir_valid))
        self.output_dtypes.append(logits.dtype)
        return logits

    def forward_with_aux(self, images, nir=None, nir_valid=None):
        features = self.features(images, nir, nir_valid)
        outputs = {"out": self.main(features),
                   "aux": {name: head(features) for name, head in self.heads.items()}}
        self.output_dtypes.extend(value.dtype for value in
                                  (outputs["out"], *outputs["aux"].values()))
        return outputs


class ObservedCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(ignore_index=255)
        self.input_dtypes = []

    def forward(self, logits, target):
        self.input_dtypes.append(logits.dtype)
        return super().forward(logits, target)


class MixedPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def configuration(mode="resize", enabled=True):
        return {"training": {"mixed_precision": enabled,
                             "mixup": {"enabled": True, "alpha": 0.2,
                                       "probability": 1.0}},
                "dataset": {"spatial_mode": mode, "input_size": 8,
                            "tile_stride": 6, "tile_batch_size": 2}}

    @staticmethod
    def batch(batch_size=2):
        rows, columns = torch.meshgrid(torch.arange(11), torch.arange(13), indexing="ij")
        target = ((rows + columns) % 2).expand(batch_size, -1, -1).clone()
        target[:, :2] = 255
        valid = (columns > 2)[None, None].expand(batch_size, 1, -1, -1).clone()
        images = torch.rand(batch_size, 3, 11, 13) - 0.5
        nir = torch.rand(batch_size, 1, 11, 13) - 0.5
        nir.masked_fill_(~valid, 0)
        return {"image": images, "mask": target, "nir": nir, "nir_valid": valid}

    def train_batch(self, device, *, enabled=True, scaler=None, model=None, optimizer=None):
        torch.manual_seed(19)
        model = TinyNIRModel().to(device) if model is None else model
        optimizer = (torch.optim.AdamW(model.parameters(), lr=0.01)
                     if optimizer is None else optimizer)
        main_loss = ObservedCrossEntropy()
        objective = SegmentationObjective(main_loss, auxiliary_weight=0.4).to(device)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with contextlib.redirect_stderr(io.StringIO()):
            statistics = train_one_epoch(model, [self.batch()], objective, optimizer,
                                         scaler, device, self.configuration(enabled=enabled))
        self.assertTrue(all(np.isfinite(value) for value in statistics.values()))
        self.assertEqual(statistics["mixup_fraction"], 1)
        self.assertEqual(set(main_loss.input_dtypes), {torch.float32})
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter).all(), name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertFalse(torch.equal(parameter.detach(), before[name]), name)
        return model, optimizer

    def test_config_enables_fp16_only_on_cuda(self):
        self.assertFalse(use_fp16("cpu", self.configuration()))
        self.assertFalse(use_fp16(torch.device("cpu"), self.configuration()))
        self.assertFalse(use_fp16("cuda", None))
        self.assertFalse(use_fp16("cuda", self.configuration(enabled=False)))
        self.assertTrue(use_fp16(torch.device("cuda:0"), self.configuration()))

    def test_cpu_fallback_trains_mixup_nir_and_ignored_labels(self):
        model, _ = self.train_batch("cpu")
        self.assertEqual(set(model.output_dtypes), {torch.float32})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 training")
    def test_cuda_fp16_trains_mixup_nir_and_all_auxiliary_heads(self):
        scaler = torch.amp.GradScaler("cuda", init_scale=128, growth_interval=1)
        model, _ = self.train_batch("cuda", scaler=scaler)
        self.assertEqual(set(model.output_dtypes), {torch.float16})
        self.assertEqual(scaler.get_scale(), 256)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 training")
    def test_cuda_can_explicitly_keep_fp32_training(self):
        model, _ = self.train_batch("cuda", enabled=False)
        self.assertEqual(set(model.output_dtypes), {torch.float32})

    def check_validation(self, device):
        torch.manual_seed(27)
        model = TinyNIRModel().to(device)
        main_loss = ObservedCrossEntropy()
        objective = SegmentationObjective(main_loss, auxiliary_weight=0.4).to(device)
        batch = self.batch(batch_size=1)
        for mode in ("resize", "crop"):
            with self.subTest(device=device, mode=mode):
                model.output_dtypes.clear()
                with contextlib.redirect_stderr(io.StringIO()):
                    value, metrics = validate(model, [batch], objective, device, 2,
                                              self.configuration(mode))
                self.assertTrue(np.isfinite(value))
                self.assertTrue(np.isfinite(metrics["ber"]))
                expected_dtype = torch.float16 if device == "cuda" else torch.float32
                self.assertEqual(set(model.output_dtypes), {expected_dtype})
                self.assertEqual(set(main_loss.input_dtypes), {torch.float32})
                # This pointwise model has the same logits with or without tiling.
                with torch.no_grad(), torch.autocast(device_type=device,
                                                    dtype=torch.float16,
                                                    enabled=device == "cuda"):
                    direct = model(batch["image"].to(device),
                                   batch["nir"].to(device), batch["nir_valid"].to(device))
                expected = objective(direct, batch["mask"].to(device)).item()
                self.assertAlmostEqual(value, expected, places=5)

    def test_cpu_fallback_validates_resize_and_native_tiles(self):
        self.check_validation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 validation")
    def test_cuda_fp16_validates_resize_and_native_tiles(self):
        self.check_validation("cuda")

    def test_half_precision_large_reductions_keep_losses_and_gradients_finite(self):
        target = (torch.arange(384 * 384).reshape(1, 384, 384) % 2).long()
        target[:, :16] = 255
        valid = torch.ones((1, 1, 384, 384), dtype=torch.bool)
        valid[:, :, :, :32] = False
        objective = SegmentationObjective(nn.CrossEntropyLoss(ignore_index=255), auxiliary_weight=0.4)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                tensors = [torch.zeros((1, 2, 384, 384), dtype=dtype, requires_grad=True)
                           for _ in range(5)]
                outputs = {"out": tensors[0],
                           "aux": dict(zip(("rgb", "hsv", "lab", "nir"), tensors[1:]))}
                loss, parts = objective(outputs, target, return_components=True,
                                        auxiliary_valid={"nir": valid})
                self.assertTrue(all(value.dtype == torch.float32 for value in parts.values()))
                self.assertAlmostEqual(loss.item(), 1.4 * np.log(2), places=5)
                loss.backward()
                for tensor in tensors:
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(torch.count_nonzero(tensor.grad).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for an enabled GradScaler")
    def test_dynamic_scaler_checkpoint_roundtrip_continues_training(self):
        scaler = torch.amp.GradScaler("cuda", init_scale=128, growth_interval=1)
        model, optimizer = self.train_batch("cuda", scaler=scaler)
        stream = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        restored = TinyNIRModel().cuda()
        restored.load_state_dict(checkpoint["model_state_dict"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restored_scaler = torch.amp.GradScaler("cuda", init_scale=2)
        restored_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())
        self.train_batch("cuda", scaler=restored_scaler, model=restored,
                         optimizer=restored_optimizer)
        self.assertEqual(restored_scaler.get_scale(), 512)


if __name__ == "__main__":
    unittest.main()

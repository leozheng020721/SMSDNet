"""Offline CPU checks for the HSV/Lab SegFormer ablation."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from color_spaces import rgb_to_hsv, rgb_to_lab
from model_factory import initialize_from_checkpoint, resolve_checkpoint_config
from model_multicolor import DualSpatialFusion, MultiColorSegFormer, PooledColorEncoder


class CheckpointConfigurationTest(unittest.TestCase):
    @staticmethod
    def runtime_config():
        return {
            "device": "cuda:1",
            "model": {"variant": "multicolor", "encoder": "mit_b3"},
            "dataset": {
                "root_dir": "E:/runtime/SEO", "type": "seo", "split_mode": "predefined",
                "val_ratio": 0.2, "spatial_mode": "crop", "input_size": 768,
                "tile_stride": 640, "tile_batch_size": 4,
            },
        }

    def test_legacy_checkpoint_restores_resize_without_stale_tile_settings(self):
        runtime = self.runtime_config()
        checkpoint = {"config": {
            "model": {"variant": "baseline", "encoder": "mit_b0"},
            "dataset": {
                "root_dir": "D:/old/AISD", "type": "aisd", "split_mode": "random",
                "input_size": 512, "normalization": {"mean": [0.4, 0.5, 0.6]},
            },
        }}
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        self.assertEqual(resolved["dataset"]["spatial_mode"], "resize")
        self.assertEqual(resolved["dataset"]["input_size"], 512)
        self.assertNotIn("tile_stride", resolved["dataset"])
        self.assertNotIn("tile_batch_size", resolved["dataset"])
        self.assertEqual(resolved["model"], checkpoint["config"]["model"])
        for name in ("root_dir", "type", "split_mode", "val_ratio"):
            self.assertEqual(resolved["dataset"][name], runtime["dataset"][name])
        self.assertEqual(resolved["device"], runtime["device"])
        self.assertEqual(runtime["dataset"]["spatial_mode"], "crop")
        self.assertEqual(runtime["dataset"]["tile_stride"], 640)
        self.assertIsNot(resolved["model"], checkpoint["config"]["model"])
        self.assertIsNot(resolved["dataset"]["normalization"], checkpoint["config"]["dataset"]["normalization"])

    def test_native_checkpoint_restores_window_parameters_and_runtime_paths(self):
        runtime = self.runtime_config()
        checkpoint = {"config": {"dataset": {
            "spatial_mode": "crop", "input_size": 512,
            "tile_stride": 384, "tile_batch_size": 2, "root_dir": "D:/old/data",
        }}}
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        for name in ("spatial_mode", "input_size", "tile_stride", "tile_batch_size"):
            self.assertEqual(resolved["dataset"][name], checkpoint["config"]["dataset"][name])
        self.assertEqual(resolved["dataset"]["root_dir"], runtime["dataset"]["root_dir"])
        self.assertEqual(runtime["dataset"]["input_size"], 768)

    def test_legacy_default_size_does_not_inherit_runtime_native_crop(self):
        for saved in ({"dataset": {}}, {"model": {"variant": "baseline"}}):
            with self.subTest(saved=saved):
                runtime = self.runtime_config()
                resolved = resolve_checkpoint_config(runtime, {"config": saved})
                self.assertEqual(resolved["dataset"]["input_size"], 512)
                self.assertEqual(resolved["dataset"]["spatial_mode"], "resize")
                self.assertEqual(runtime["dataset"]["input_size"], 768)

    def test_unspecified_native_window_settings_do_not_leak_from_runtime(self):
        checkpoint = {"config": {"dataset": {"spatial_mode": "crop", "input_size": 512}}}
        resolved = resolve_checkpoint_config(self.runtime_config(), checkpoint)
        self.assertEqual(resolved["dataset"]["spatial_mode"], "crop")
        self.assertNotIn("tile_stride", resolved["dataset"])
        self.assertNotIn("tile_batch_size", resolved["dataset"])

    def test_same_seo_checkpoint_restores_group_split_and_keeps_runtime_paths(self):
        runtime = self.runtime_config()
        runtime["random_seed"] = 99
        runtime["dataset"]["val_ratio"] = 0.5
        checkpoint = {"config": {
            "random_seed": 42,
            "dataset": {"type": "SEO", "val_ratio": 0.3, "root_dir": "D:/old/SEO"},
        }}
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        self.assertEqual(resolved["random_seed"], 42)
        self.assertEqual(resolved["dataset"]["val_ratio"], 0.3)
        for name in ("type", "root_dir", "split_mode"):
            self.assertEqual(resolved["dataset"][name], runtime["dataset"][name])
        self.assertEqual(resolved["device"], runtime["device"])
        self.assertEqual(runtime["random_seed"], 99)
        self.assertEqual(runtime["dataset"]["val_ratio"], 0.5)

    def test_seo_checkpoint_keeps_aisd_runtime_split_parameters(self):
        runtime = self.runtime_config()
        runtime["random_seed"] = 99
        runtime["dataset"].update(type="aisd", root_dir="E:/runtime/AISD", val_ratio=0.5)
        checkpoint = {"config": {
            "random_seed": 42,
            "dataset": {"type": "seo", "val_ratio": 0.2, "root_dir": "D:/old/SEO"},
        }}
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        self.assertEqual(resolved["random_seed"], 99)
        for name in ("type", "root_dir", "split_mode", "val_ratio"):
            self.assertEqual(resolved["dataset"][name], runtime["dataset"][name])
        self.assertEqual(resolved["device"], runtime["device"])

    def test_same_seo_checkpoint_restores_default_ratio_when_not_saved(self):
        runtime = self.runtime_config()
        runtime["random_seed"] = 99
        runtime["dataset"]["val_ratio"] = 0.5
        checkpoint = {"config": {"dataset": {"type": "seo"}}}
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        self.assertEqual(resolved["dataset"]["val_ratio"], 0.2)
        self.assertEqual(resolved["random_seed"], 99)
        self.assertEqual(runtime["dataset"]["val_ratio"], 0.5)


class ColorSpaceTest(unittest.TestCase):
    def test_conversions_match_opencv_float_reference(self):
        random = np.random.default_rng(7)
        rgb = random.random((17, 19, 3), dtype=np.float32)
        rgb[0, :6] = np.array([
            [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        hsv_reference = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv_reference[..., 0] /= 360.0
        lab_reference = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        lab_reference[..., 0] /= 100.0
        lab_reference[..., 1:] = (lab_reference[..., 1:] + 128.0) / 255.0

        hsv = rgb_to_hsv(tensor)[0].permute(1, 2, 0).numpy()
        lab = rgb_to_lab(tensor)[0].permute(1, 2, 0).numpy()

        np.testing.assert_allclose(hsv, hsv_reference, rtol=0, atol=2e-6)
        # OpenCV uses an interpolation table for its float Lab conversion.
        np.testing.assert_allclose(lab, lab_reference, rtol=0, atol=3e-3)

    def test_black_and_gray_have_finite_outputs_and_gradients(self):
        levels = torch.tensor([0.0, 1e-8, 0.04, 0.5, 1.0])
        rgb = levels.reshape(1, 1, 1, -1).repeat(1, 3, 2, 1).requires_grad_()

        hsv, lab = rgb_to_hsv(rgb), rgb_to_lab(rgb)
        (hsv.square().mean() + lab.square().mean()).backward()

        self.assertTrue(torch.isfinite(hsv).all())
        self.assertTrue(torch.isfinite(lab).all())
        self.assertTrue(torch.isfinite(rgb.grad).all())
        torch.testing.assert_close(hsv[:, :2], torch.zeros_like(hsv[:, :2]))


class MultiColorModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(9)

    @staticmethod
    def make_model(**kwargs):
        return MultiColorSegFormer(
            encoder_name="mit_b0", encoder_weights=None,
            decoder_segmentation_channels=32, in_channels=3, classes=2,
            aux_channels=(4, 8, 12, 16), gate_channels=4, **kwargs,
        )

    def test_two_spatial_gates_are_independent_and_bound_residual(self):
        fusion = DualSpatialFusion(8, 4, gate_channels=4)
        rgb = torch.randn(2, 8, 8, 10)
        auxiliary = torch.randn(2, 4, 4, 5)
        with torch.no_grad():
            fusion.gate[2].weight.zero_()
            fusion.gate[2].bias.copy_(torch.logit(torch.tensor([0.2, 0.3])))

        fused, gates = fusion(rgb, auxiliary, auxiliary)

        self.assertEqual(fused.shape, rgb.shape)
        self.assertEqual(gates.shape, (2, 2, 8, 10))
        torch.testing.assert_close(gates[:, 0], torch.full_like(gates[:, 0], 0.2))
        torch.testing.assert_close(gates[:, 1], torch.full_like(gates[:, 1], 0.3))
        self.assertFalse(torch.allclose(gates.sum(dim=1), torch.ones_like(gates[:, 0])))
        zeros = torch.zeros_like(auxiliary)
        torch.testing.assert_close(fusion(rgb, zeros, zeros)[0], rgb)
        with torch.no_grad():
            fusion.gate[2].bias.fill_(-100.0)
        torch.testing.assert_close(fusion(rgb, auxiliary, auxiliary)[0], rgb)

    def test_average_and_max_pooling_produce_four_scales(self):
        average = PooledColorEncoder((4, 8, 12, 16), pooling="avg")
        maximum = PooledColorEncoder((4, 8, 12, 16), pooling="max")
        maximum.load_state_dict(average.state_dict())
        x = torch.randn(1, 3, 64, 64)
        average_features = average(x)
        maximum_features = maximum(x)

        for index, (avg, max_) in enumerate(zip(average_features, maximum_features)):
            self.assertEqual(avg.shape, (1, (4, 8, 12, 16)[index], 16 // 2**index, 16 // 2**index))
            self.assertEqual(max_.shape, avg.shape)
        self.assertFalse(torch.allclose(average_features[0], maximum_features[0]))

    def test_real_mit_forward_backward_and_unmodified_rgb_stages(self):
        model = self.make_model().eval()
        raw = torch.rand(1, 3, 64, 64)
        x = (raw - model.input_mean) / model.input_std
        with torch.no_grad():
            original = model.encoder(x)
            features = model.forward_features(x)
        for index, stage_index in enumerate(model.stage_indices):
            torch.testing.assert_close(features["rgb"][index], original[stage_index])
            self.assertEqual(features["fused"][index].shape, original[stage_index].shape)
            self.assertEqual(features["gates"][index].shape, (1, 2, 16 // 2**index, 16 // 2**index))
            self.assertTrue(((features["gates"][index] >= 0) & (features["gates"][index] <= 1)).all())
            torch.testing.assert_close(features["decoder_features"][stage_index], features["fused"][index])

        prediction = model(x)
        self.assertEqual(prediction.shape, (1, 2, 64, 64))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        for module in (model.encoder, model.hsv_encoder, model.lab_encoder, *model.fusions):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
        for fusion in model.fusions:
            self.assertGreater(fusion.gate[2].weight.grad.abs().sum().item(), 0)
        self.assertNotEqual(
            model.hsv_encoder.stages[0][1].weight.data_ptr(),
            model.lab_encoder.stages[0][1].weight.data_ptr(),
        )

    def test_auxiliary_colors_restore_configured_rgb_normalization(self):
        model = self.make_model(input_mean=(0.2, 0.3, 0.4), input_std=(0.3, 0.4, 0.5)).eval()
        raw = torch.rand(1, 3, 64, 64)
        x = (raw - model.input_mean) / model.input_std
        captured = {}

        def capture(name):
            def hook(module, inputs):
                captured[name] = inputs[0].detach().clone()
            return hook

        handles = [
            model.hsv_encoder.register_forward_pre_hook(capture("hsv")),
            model.lab_encoder.register_forward_pre_hook(capture("lab")),
        ]
        try:
            with torch.no_grad():
                model(x)
        finally:
            for handle in handles:
                handle.remove()

        torch.testing.assert_close(captured["hsv"], rgb_to_hsv(raw) * 2 - 1, atol=2e-6, rtol=0)
        torch.testing.assert_close(captured["lab"], rgb_to_lab(raw) * 2 - 1, atol=2e-6, rtol=0)

    def test_auxiliary_predictions_use_independent_heads_and_one_encoding(self):
        model = self.make_model(auxiliary_supervision=True, aux_decoder_channels=8).eval()
        raw = torch.rand(1, 3, 64, 96)
        x = (raw - model.input_mean) / model.input_std
        calls = {"rgb": 0, "hsv": 0, "lab": 0}

        def count(name):
            def hook(module, inputs, output):
                calls[name] += 1
            return hook

        handles = [
            module.register_forward_hook(count(name))
            for name, module in (("rgb", model.encoder), ("hsv", model.hsv_encoder),
                                 ("lab", model.lab_encoder))
        ]
        try:
            with torch.no_grad():
                outputs = model.forward_with_aux(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {"rgb": 1, "hsv": 1, "lab": 1})
        self.assertEqual(set(outputs["aux"]), {"rgb", "hsv", "lab"})
        for prediction in (outputs["out"], *outputs["aux"].values()):
            self.assertEqual(prediction.shape, (1, 2, 64, 96))
            self.assertTrue(torch.isfinite(prediction).all())
        parameter_ids = [
            {id(parameter) for parameter in head.parameters()}
            for head in model.auxiliary_heads.values()
        ]
        for index, parameters in enumerate(parameter_ids):
            self.assertTrue(all(parameters.isdisjoint(other) for other in parameter_ids[index + 1:]))
        with torch.no_grad():
            default_output = model(x)
        self.assertIsInstance(default_output, torch.Tensor)
        torch.testing.assert_close(default_output, outputs["out"])

    def test_auxiliary_losses_directly_supervise_each_native_encoder(self):
        model = self.make_model(auxiliary_supervision=True, aux_decoder_channels=8).eval()
        raw = torch.rand(1, 3, 64, 64)
        x = (raw - model.input_mean) / model.input_std
        target = torch.randint(0, 2, (1, 64, 64))
        encoders = {"rgb": model.encoder, "hsv": model.hsv_encoder, "lab": model.lab_encoder}
        for name, encoder in encoders.items():
            with self.subTest(branch=name):
                model.zero_grad(set_to_none=True)
                outputs = model.forward_with_aux(x)
                torch.nn.functional.cross_entropy(outputs["aux"][name], target).backward()
                gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
                for other_name, other_encoder in encoders.items():
                    if other_name != name:
                        self.assertTrue(all(parameter.grad is None for parameter in other_encoder.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in model.fusions.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))

    def test_legacy_model_keeps_strict_checkpoint_compatibility(self):
        legacy = self.make_model()
        model = self.make_model(auxiliary_supervision=False)
        self.assertFalse(hasattr(model, "auxiliary_heads"))
        model.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "auxiliary_supervision"):
            model.forward_with_aux(torch.rand(1, 3, 64, 64))

    def test_baseline_weights_transfer_without_overwriting_new_branches(self):
        baseline = smp.Segformer(
            encoder_name="mit_b0", encoder_weights=None,
            decoder_segmentation_channels=32, in_channels=3, classes=2,
        )
        model = self.make_model()
        original_auxiliary = model.hsv_encoder.stages[0][1].weight.detach().clone()
        state = baseline.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "baseline.pth"
            torch.save({"model_state_dict": {"module." + key: value for key, value in state.items()}}, checkpoint)
            report = initialize_from_checkpoint(model, checkpoint)

        self.assertEqual(report["loaded"], len(state))
        self.assertEqual(report["skipped"], 0)
        self.assertGreater(report["new"], 0)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        torch.testing.assert_close(model.hsv_encoder.stages[0][1].weight, original_auxiliary)

    def test_multicolor_transfer_preserves_branches_and_leaves_auxiliary_heads_fresh(self):
        source = self.make_model(auxiliary_supervision=True, aux_decoder_channels=8)
        model = self.make_model(auxiliary_supervision=True, aux_decoder_channels=8)
        original_heads = {key: value.detach().clone() for key, value in model.auxiliary_heads.state_dict().items()}
        source_state = source.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "multicolor.pth"
            for include_heads in (False, True):
                with self.subTest(checkpoint_has_auxiliary_heads=include_heads):
                    state = {
                        key: value for key, value in source_state.items()
                        if include_heads or not key.startswith("auxiliary_heads.")
                    }
                    torch.save({"model_state_dict": state}, checkpoint)
                    report = initialize_from_checkpoint(model, checkpoint)
                    transferred = {key: value for key, value in state.items() if not key.startswith("auxiliary_heads.")}
                    self.assertEqual(report["loaded"], len(transferred))
                    self.assertEqual(report["skipped"], len(state) - len(transferred))
                    for key, value in transferred.items():
                        torch.testing.assert_close(model.state_dict()[key], value)
                    for key, value in original_heads.items():
                        torch.testing.assert_close(model.auxiliary_heads.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()

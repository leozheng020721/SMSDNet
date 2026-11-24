"""CPU checks for the optional NIR branch and legacy checkpoint compatibility."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from model_factory import build_model, initialize_from_checkpoint, resolve_checkpoint_config
from model_multicolor import MultiColorSegFormer


class NIRModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(73)

    @staticmethod
    def make_model(**options):
        return MultiColorSegFormer(
            encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=2,
            decoder_segmentation_channels=32, aux_channels=(4, 8, 12, 16),
            gate_channels=4, auxiliary_supervision=True, aux_decoder_channels=8,
            **options,
        )

    @staticmethod
    def config(enabled=True):
        return {
            "device": "cpu",
            "model": {
                "variant": "multicolor", "encoder": "mit_b0", "encoder_weights": None,
                "multicolor": {
                    "aux_channels": [4, 8, 12, 16], "gate_channels": 4,
                    "auxiliary_supervision": True, "aux_decoder_channels": 8,
                    "nir": {"enabled": enabled},
                },
            },
            "dataset": {
                "type": "seo", "root_dir": "E:/runtime/SEO", "input_size": 64,
                "in_channels": 3, "num_classes": 2,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "nir": {
                    "enabled": enabled, "root_dir": "E:/runtime/nir",
                    "manifest": "runtime.csv", "include_candidates": False,
                    "normalization": {"percentiles": [1.0, 99.0]}, "crop_retries": 4,
                },
            },
        }

    def test_forward_has_four_independent_native_heads_and_one_nir_encoding(self):
        model = self.make_model(nir_enabled=True).eval()
        x, nir = torch.randn(1, 3, 64, 96), torch.rand(1, 1, 64, 96) * 2 - 1
        observed = []
        handle = model.nir_encoder.register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0].detach().clone())
        )
        try:
            with torch.no_grad():
                output = model.forward_with_aux(x, nir)
        finally:
            handle.remove()
        self.assertTrue(model.nir_enabled)
        self.assertEqual(len(observed), 1)
        torch.testing.assert_close(observed[0], nir, rtol=0, atol=0)
        self.assertEqual(set(output["aux"]), {"rgb", "hsv", "lab", "nir"})
        for logits in (output["out"], *output["aux"].values()):
            self.assertEqual(logits.shape, (1, 2, 64, 96))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(model.nir_encoder.stages[0][1].in_channels, 1)
        self.assertEqual(model.hsv_encoder.stages[0][1].in_channels, 3)
        parameters = [{id(p) for p in head.parameters()} for head in model.auxiliary_heads.values()]
        for index, branch in enumerate(parameters):
            self.assertTrue(all(branch.isdisjoint(other) for other in parameters[index + 1:]))
        with torch.no_grad():
            torch.testing.assert_close(model(x, nir), output["out"])

    def test_all_invalid_nir_is_exactly_legacy_fusion_even_with_cca(self):
        options = {"cross_scale_attention": True, "cross_scale_attention_channels": 4}
        legacy = self.make_model(**options).eval()
        model = self.make_model(nir_enabled=True, **options).eval()
        missing = model.load_state_dict(legacy.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(name.startswith(("nir_", "auxiliary_heads.nir.")) for name in missing.missing_keys))
        with torch.no_grad():
            legacy.cross_scale_attention.gamma.fill_(0.5)
            model.cross_scale_attention.gamma.fill_(0.5)
            x = torch.randn(1, 3, 64, 64)
            valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
            expected = legacy.forward_with_aux(x)
            output_a = model.forward_with_aux(x, torch.ones_like(valid, dtype=torch.float32), valid)
            output_b = model.forward_with_aux(x, torch.randn(1, 1, 64, 64) * 100, valid)
            features = model.forward_features(x, torch.randn(1, 1, 64, 64), valid)
        torch.testing.assert_close(output_a["out"], expected["out"], rtol=0, atol=0)
        torch.testing.assert_close(output_a["out"], output_b["out"], rtol=0, atol=0)
        for name in expected["aux"]:
            torch.testing.assert_close(output_a["aux"][name], expected["aux"][name], rtol=0, atol=0)
        for feature, gate in zip(features["nir"], features["nir_gates"]):
            self.assertEqual(torch.count_nonzero(feature).item(), 0)
            self.assertEqual(torch.count_nonzero(gate).item(), 0)

    def test_partial_masks_prevent_bias_or_invalid_values_leaking_into_fusion(self):
        model = self.make_model(nir_enabled=True, pooling="max").eval()
        legacy = self.make_model(pooling="max").eval()
        legacy.load_state_dict({name: value for name, value in model.state_dict().items()
                                if not name.startswith(("nir_", "auxiliary_heads.nir."))}, strict=True)
        x, nir = torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
        valid = torch.ones_like(nir, dtype=torch.bool)
        valid[:, :, :16, :] = False
        valid[:, :, 32, 32] = False
        changed = nir.masked_fill(~valid, 10000)
        with torch.no_grad():
            for stage in model.nir_encoder.stages:
                stage[1].bias.fill_(2.0)
            original = model.forward_features(x, nir, valid)
            modified = model.forward_features(x, changed, valid[:, 0].float())
            base = legacy.forward_features(x)
        expected_mask = valid
        for index in range(4):
            stride = 4 if index == 0 else 2
            expected_mask = F.avg_pool2d(expected_mask.float(), stride, stride) >= 1
            torch.testing.assert_close(original["nir_valid_features"][index], expected_mask)
            for key in ("nir", "fused", "gates", "nir_gates"):
                torch.testing.assert_close(original[key][index], modified[key][index], rtol=0, atol=0)
            invalid = ~expected_mask
            self.assertTrue((original["nir"][index].masked_select(invalid) == 0).all())
            self.assertTrue((original["nir_gates"][index].masked_select(invalid) == 0).all())
            torch.testing.assert_close(original["fused"][index].masked_select(invalid),
                                       base["fused"][index].masked_select(invalid), rtol=0, atol=0)
            for key in ("rgb", "hsv", "lab", "gates"):
                torch.testing.assert_close(original[key][index], base[key][index], rtol=0, atol=0)

    def test_missing_malformed_nonfinite_and_nonfloat_nir_are_rejected(self):
        model = self.make_model(nir_enabled=True)
        x = torch.randn(1, 3, 64, 64)
        cases = [
            (None, "explicit nir"),
            (torch.zeros(1, 64, 64), "B1HW"),
            (torch.zeros(1, 3, 64, 64), "B1HW"),
            (torch.zeros(2, 1, 64, 64), "B1HW"),
            (torch.zeros(1, 1, 32, 64), "B1HW"),
            (torch.full((1, 1, 64, 64), float("nan")), "finite"),
            (torch.full((1, 1, 64, 64), float("inf")), "finite"),
            (torch.zeros(1, 1, 64, 64, dtype=torch.int64), "floating-point"),
        ]
        for nir, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model(x, nir)
        with self.assertRaisesRegex(ValueError, "finite"):
            model(x, cases[5][0], torch.zeros(1, 1, 64, 64, dtype=torch.bool))

    def test_malformed_or_nonfinite_masks_are_rejected(self):
        model = self.make_model(nir_enabled=True)
        x, nir = torch.randn(1, 3, 64, 64), torch.zeros(1, 1, 64, 64)
        for mask, message in (
            (torch.ones(1, 2, 64, 64), "shape"),
            (torch.ones(1, 64, 32), "shape"),
            (torch.full_like(nir, float("nan")), "finite"),
            (torch.full_like(nir, 2), r"\[0, 1\]"),
            (torch.ones_like(nir, dtype=torch.int64), "bool or floating"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model(x, nir, mask)

    def test_main_loss_trains_nir_encoder_and_all_nir_fusion_stages(self):
        model = self.make_model(nir_enabled=True).eval()
        x, nir = torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 64, 64))
        F.cross_entropy(model(x, nir), target).backward()
        for module in (model.nir_encoder, *model.nir_fusions):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.auxiliary_heads["nir"].parameters()))

    def test_nir_auxiliary_loss_supervises_only_its_native_encoder_and_head(self):
        model = self.make_model(nir_enabled=True).eval()
        output = model.forward_with_aux(torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64))
        F.cross_entropy(output["aux"]["nir"], torch.randint(0, 2, (1, 64, 64))).backward()
        for module in (model.nir_encoder, model.auxiliary_heads["nir"]):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        for module in (model.encoder, model.hsv_encoder, model.lab_encoder,
                       model.fusions, model.nir_fusions, model.decoder):
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_default_off_has_no_new_state_and_strictly_loads_old_three_branch_state(self):
        torch.manual_seed(81)
        legacy = self.make_model()
        torch.manual_seed(81)
        disabled = self.make_model(nir_enabled=False)
        self.assertFalse(disabled.nir_enabled)
        self.assertFalse(hasattr(disabled, "nir_encoder"))
        self.assertEqual(set(disabled.auxiliary_heads), {"rgb", "hsv", "lab"})
        self.assertFalse(any(name.startswith(("nir_", "auxiliary_heads.nir."))
                             for name in disabled.state_dict()))
        disabled.load_state_dict(legacy.state_dict(), strict=True)
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(disabled.state_dict()[name], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "nir_enabled"):
            disabled(torch.randn(1, 3, 64, 64), torch.zeros(1, 1, 64, 64))

    def test_warm_start_transfers_old_gates_and_can_transfer_new_nir_modules(self):
        source = self.make_model()
        target = self.make_model(nir_enabled=True)
        original = {name: value.clone() for name, value in target.nir_encoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"model_state_dict": source.state_dict()}, path)
            report = initialize_from_checkpoint(target, path, include_auxiliary_heads=True)
            self.assertEqual(report["loaded"], len(source.state_dict()))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value)
            for name, value in original.items():
                torch.testing.assert_close(target.nir_encoder.state_dict()[name], value)
            other = self.make_model(nir_enabled=True)
            torch.save({"model_state_dict": target.state_dict()}, path)
            report = initialize_from_checkpoint(other, path, include_auxiliary_heads=True)
        self.assertEqual(report["loaded"], len(target.state_dict()))
        self.assertEqual(report["new"], 0)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(other.state_dict()[name], value)

    def test_factory_enforces_separate_rgb_and_nir_flags(self):
        model = build_model(self.config())
        self.assertTrue(model.nir_enabled)
        self.assertEqual(model.nir_encoder.stages[0][1].in_channels, 1)
        self.assertEqual(model.encoder.patch_embed1.proj.in_channels, 3)
        for field in ("model", "dataset"):
            config = self.config()
            section = config["model"]["multicolor"] if field == "model" else config["dataset"]
            section["nir"]["enabled"] = False
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "must agree"):
                build_model(config)
        config = self.config()
        config["model"]["variant"] = "baseline"
        with self.assertRaisesRegex(ValueError, "multicolor"):
            build_model(config)
        config = self.config()
        config["dataset"]["in_channels"] = 4
        with self.assertRaisesRegex(ValueError, "three RGB"):
            build_model(config)
        config = self.config()
        config["dataset"]["nir"]["enabled"] = "false"
        with self.assertRaisesRegex(ValueError, "booleans"):
            build_model(config)

    def test_checkpoint_restores_nir_science_and_keeps_runtime_nir_paths(self):
        saved = self.config()
        saved["dataset"]["nir"].update(
            root_dir="D:/saved/nir", manifest="saved.csv", include_candidates=True,
            normalization={"percentiles": [2.0, 98.0]}, crop_retries=7,
        )
        runtime = self.config(False)
        before = copy.deepcopy(runtime)
        resolved = resolve_checkpoint_config(runtime, {"config": saved})
        self.assertTrue(resolved["model"]["multicolor"]["nir"]["enabled"])
        for name in ("root_dir", "manifest"):
            self.assertEqual(resolved["dataset"]["nir"][name], runtime["dataset"]["nir"][name])
        for name in ("enabled", "include_candidates", "normalization", "crop_retries"):
            self.assertEqual(resolved["dataset"]["nir"][name], saved["dataset"]["nir"][name])
        self.assertIsNot(resolved["dataset"]["nir"]["normalization"],
                         saved["dataset"]["nir"]["normalization"])
        self.assertEqual(runtime, before)

    def test_legacy_checkpoint_does_not_inherit_runtime_nir_enablement(self):
        saved = self.config(False)
        del saved["model"]["multicolor"]["nir"]
        del saved["dataset"]["nir"]
        runtime = self.config(True)
        resolved = resolve_checkpoint_config(runtime, {"config": saved})
        self.assertFalse(resolved["dataset"]["nir"]["enabled"])
        self.assertEqual(resolved["dataset"]["nir"]["root_dir"], runtime["dataset"]["nir"]["root_dir"])
        self.assertNotIn("nir", resolved["model"]["multicolor"])
        self.assertFalse(build_model(resolved).nir_enabled)


if __name__ == "__main__":
    unittest.main()

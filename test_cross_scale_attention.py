"""Offline CPU checks for cross-scale criss-cross attention and model wiring."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from cross_scale_attention import CrossScaleCrissCrossAttention
from model_factory import build_model, initialize_from_checkpoint, resolve_checkpoint_config
from model_multicolor import MultiColorSegFormer


def dense_restricted_attention(module, query_feature, context_feature):
    """Reference over all image positions, with a Boolean row/column union."""
    height, width = query_feature.shape[-2:]
    context = F.interpolate(
        context_feature, size=(height, width), mode="bilinear", align_corners=False
    )
    query = module.query_projection(query_feature).flatten(2)
    key = module.key_projection(context).flatten(2)
    value = module.value_projection(context).flatten(2)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=query.device),
        torch.arange(width, device=query.device),
        indexing="ij",
    )
    rows, columns = rows.flatten(), columns.flatten()
    allowed = (rows[:, None] == rows[None, :]) | (columns[:, None] == columns[None, :])
    scores = torch.bmm(query.transpose(1, 2), key) / math.sqrt(query.shape[1])
    weights = scores.masked_fill(~allowed[None], float("-inf")).softmax(dim=-1)
    aggregated = torch.bmm(value, weights.transpose(1, 2))
    aggregated = aggregated.reshape(query_feature.shape[0], value.shape[1], height, width)
    return query_feature + module.gamma * module.out_projection(aggregated)


class CPUAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_matches_dense_union_reference_for_rectangles_and_degenerate_axes(self):
        torch.manual_seed(41)
        shapes = ((3, 5, 2, 3), (1, 5, 1, 3), (5, 1, 3, 1), (1, 1, 1, 1))
        for height, width, context_height, context_width in shapes:
            with self.subTest(query=(height, width), context=(context_height, context_width)):
                module = CrossScaleCrissCrossAttention(
                    3, 5, attention_channels=2, value_channels=4
                ).double().eval()
                with torch.no_grad():
                    module.gamma.fill_(0.7)
                    query = torch.randn(2, 3, height, width, dtype=torch.float64)
                    context = torch.randn(2, 5, context_height, context_width, dtype=torch.float64)
                    actual = module(query, context)
                    expected = dense_restricted_attention(module, query, context)
                self.assertEqual(actual.shape, query.shape)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_zero_gamma_starts_as_exact_identity(self):
        module = CrossScaleCrissCrossAttention(3, 5, attention_channels=2)
        query = torch.randn(2, 3, 3, 7)
        context = torch.randn(2, 5, 2, 4)
        self.assertEqual(module.gamma.item(), 0.0)
        torch.testing.assert_close(module(query, context), query, rtol=0, atol=0)

    def test_nonzero_gamma_propagates_finite_nonzero_query_key_value_gradients(self):
        torch.manual_seed(17)
        module = CrossScaleCrissCrossAttention(3, 5, attention_channels=2, value_channels=4)
        with torch.no_grad():
            module.gamma.fill_(0.5)
        query = torch.randn(2, 3, 3, 5, requires_grad=True)
        context = torch.randn(2, 5, 2, 3, requires_grad=True)
        module(query, context).square().mean().backward()
        for name in ("query_projection", "key_projection", "value_projection", "out_projection"):
            with self.subTest(projection=name):
                gradient = getattr(module, name).weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.0)
        for gradient in (query.grad, context.grad, module.gamma.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


class CrossScaleModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_model(**options):
        return MultiColorSegFormer(
            encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=2,
            aux_channels=(4, 8, 12, 16), gate_channels=4,
            auxiliary_supervision=True, aux_decoder_channels=8, **options,
        )

    @staticmethod
    def model_config(enabled):
        return {
            "device": "cpu",
            "random_seed": 42,
            "model": {
                "variant": "multicolor", "encoder": "mit_b0", "encoder_weights": None,
                "multicolor": {
                    "aux_channels": [4, 8, 12, 16], "gate_channels": 4,
                    "auxiliary_supervision": True, "aux_decoder_channels": 8,
                    "cross_scale_attention": {"enabled": enabled, "attention_channels": 4},
                },
            },
            "dataset": {
                "type": "seo", "root_dir": "E:/runtime/SEO", "val_ratio": 0.2,
                "in_channels": 3, "num_classes": 2, "spatial_mode": "crop",
                "input_size": 512, "tile_stride": 384, "tile_batch_size": 4,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                },
            },
        }

    def test_legacy_default_state_loads_strictly_when_attention_is_disabled(self):
        legacy = self.make_model()
        legacy_config = self.model_config(enabled=False)
        del legacy_config["model"]["multicolor"]["cross_scale_attention"]
        explicit_disabled = build_model(legacy_config)
        self.assertIsNone(explicit_disabled.cross_scale_attention)
        self.assertFalse(any(name.startswith("cross_scale_attention.") for name in legacy.state_dict()))
        explicit_disabled.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(set(legacy.state_dict()), set(explicit_disabled.state_dict()))

    def test_enabled_model_changes_only_fused_stage_three_and_preserves_auxiliary_heads(self):
        torch.manual_seed(23)
        baseline = self.make_model().eval()
        enhanced = self.make_model(
            cross_scale_attention=True, cross_scale_attention_channels=4
        ).eval()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            enhanced.load_state_dict(baseline.state_dict(), strict=True)
        incompatible = enhanced.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(name.startswith("cross_scale_attention.") for name in incompatible.missing_keys))
        inputs = torch.randn(1, 3, 64, 96)
        with torch.no_grad():
            enhanced.cross_scale_attention.gamma.fill_(0.5)
            original = baseline.forward_features(inputs)
            modified = enhanced.forward_features(inputs)
            expected_stage_three = dense_restricted_attention(
                enhanced.cross_scale_attention, original["fused"][2], original["fused"][3]
            )
            torch.testing.assert_close(modified["fused"][2], expected_stage_three)
            self.assertGreater((modified["fused"][2] - original["fused"][2]).abs().sum().item(), 0.0)
            for index in (0, 1, 3):
                torch.testing.assert_close(modified["fused"][index], original["fused"][index])
            for index, stage_index in enumerate(enhanced.stage_indices):
                torch.testing.assert_close(modified["decoder_features"][stage_index], modified["fused"][index])
            original_output = baseline.forward_with_aux(inputs)
            enhanced_output = enhanced.forward_with_aux(inputs)
            torch.testing.assert_close(enhanced(inputs), enhanced_output["out"])
            self.assertEqual(tuple(enhanced_output["out"].shape), (1, 2, 64, 96))
            for name in ("rgb", "hsv", "lab"):
                torch.testing.assert_close(enhanced_output["aux"][name], original_output["aux"][name])

    def test_training_supervises_attention_and_all_native_color_heads(self):
        torch.manual_seed(31)
        model = self.make_model(
            cross_scale_attention=True, cross_scale_attention_channels=4
        ).train()
        with torch.no_grad():
            model.cross_scale_attention.gamma.fill_(0.5)
        outputs = model.forward_with_aux(torch.randn(2, 3, 64, 64))
        target = torch.randint(0, 2, (2, 64, 64))
        auxiliary = torch.stack([F.cross_entropy(logits, target) for logits in outputs["aux"].values()]).mean()
        loss = F.cross_entropy(outputs["out"], target) + 0.4 * auxiliary
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ("query_projection", "key_projection", "value_projection", "out_projection"):
            gradient = getattr(model.cross_scale_attention, name).weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        for module in (model.encoder, model.hsv_encoder, model.lab_encoder, *model.auxiliary_heads.values()):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

    def test_new_checkpoint_restores_attention_config_and_loads_strictly(self):
        config = self.model_config(enabled=True)
        original = build_model(config).eval()
        with torch.no_grad():
            original.cross_scale_attention.gamma.fill_(0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross_scale.pth"
            torch.save({"config": config, "model_state_dict": original.state_dict()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        runtime = self.model_config(enabled=False)
        runtime["dataset"]["root_dir"] = "E:/relocated/SEO"
        resolved = resolve_checkpoint_config(runtime, checkpoint)
        restored = build_model(resolved).eval()
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertIsInstance(restored.cross_scale_attention, CrossScaleCrissCrossAttention)
        self.assertEqual(restored.cross_scale_attention.query_projection.out_channels, 4)
        self.assertEqual(resolved["dataset"]["root_dir"], runtime["dataset"]["root_dir"])
        self.assertFalse(runtime["model"]["multicolor"]["cross_scale_attention"]["enabled"])
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), original(inputs))

    def test_opt_in_auxiliary_warm_start_keeps_new_attention_fresh_and_optimizer_unrestored(self):
        source = self.make_model()
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=9e-5)
        parameter = next(source.parameters())
        parameter.grad = torch.ones_like(parameter)
        source_optimizer.step()
        self.assertTrue(source_optimizer.state)
        target = self.make_model(cross_scale_attention=True, cross_scale_attention_channels=4)
        optimizer = torch.optim.AdamW(target.parameters(), lr=7e-4)
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        fresh_attention = {name: value.clone() for name, value in target.cross_scale_attention.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_auxiliary.pth"
            torch.save({
                "model_state_dict": source.state_dict(),
                "optimizer_state_dict": source_optimizer.state_dict(), "epoch": 19,
            }, path)
            result = initialize_from_checkpoint(target, path, include_auxiliary_heads=True)
        self.assertEqual(result["loaded"], len(source.state_dict()))
        self.assertNotIn("epoch", result)
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value)
        for name, value in target.cross_scale_attention.state_dict().items():
            torch.testing.assert_close(value, fresh_attention[name])


if __name__ == "__main__":
    unittest.main()

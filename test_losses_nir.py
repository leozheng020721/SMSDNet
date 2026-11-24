"""CPU checks for masked NIR supervision and validity-aware batch MixUp."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from losses import FocalDiceLoss, SegmentationObjective
from mixup import BatchMixUp, mixup_objective


class _CpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)


class NirObjectiveTests(_CpuTests):
    def setUp(self):
        torch.manual_seed(19)
        self.outputs = {"out": torch.randn(2, 2, 2, 3, requires_grad=True),
                        "aux": {name: torch.randn(2, 2, 2, 3, requires_grad=True)
                                for name in ("rgb", "hsv", "lab", "nir")}}
        self.target = torch.tensor([[[0, 1, 255], [1, 0, 1]],
                                    [[1, 0, 1], [0, 1, 0]]])
        self.valid = torch.tensor([[[[True, False, True], [True, False, True]]],
                                   [[[False, True, True], [False, True, False]]]])
        self.objective = SegmentationObjective(nn.CrossEntropyLoss(ignore_index=255), 255, 0.4)

    def test_four_auxiliary_losses_are_averaged_without_mutating_targets(self):
        original_target, original_valid = self.target.clone(), self.valid.clone()
        total, parts = self.objective(self.outputs, self.target, return_components=True,
                                     auxiliary_valid={"nir": self.valid})
        expected = [F.cross_entropy(self.outputs["aux"][name], self.target, ignore_index=255)
                    for name in ("rgb", "hsv", "lab")]
        nir_target = self.target.masked_fill(~self.valid[:, 0], 255)
        expected.append(F.cross_entropy(self.outputs["aux"]["nir"], nir_target, ignore_index=255))
        torch.testing.assert_close(parts["aux"], torch.stack(expected).mean())
        torch.testing.assert_close(total, parts["main"] + 0.4 * torch.stack(expected).mean())
        torch.testing.assert_close(self.target, original_target)
        torch.testing.assert_close(self.valid, original_valid)
        total.backward()
        for logits in (self.outputs["out"], *self.outputs["aux"].values()):
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_invalid_and_ignored_nir_logits_cannot_change_loss_or_gradient(self):
        excluded = (~self.valid[:, 0]) | (self.target == 255)
        first = self.outputs["aux"]["nir"]
        changed = first.detach().clone()
        changed[:, 0].masked_fill_(excluded, 10000)
        changed[:, 1].masked_fill_(excluded, -10000)
        changed.requires_grad_(True)
        first_loss = self.objective(self.outputs, self.target, auxiliary_valid={"nir": self.valid})
        second_outputs = {**self.outputs, "aux": {**self.outputs["aux"], "nir": changed}}
        second_loss = self.objective(second_outputs, self.target, auxiliary_valid={"nir": self.valid})
        torch.testing.assert_close(first_loss, second_loss)
        first_gradient, = torch.autograd.grad(first_loss, first)
        second_gradient, = torch.autograd.grad(second_loss, changed)
        torch.testing.assert_close(first_gradient, second_gradient)
        self.assertEqual(first_gradient.masked_select(excluded[:, None]).abs().sum().item(), 0)
        self.assertGreater(first_gradient.masked_select(~excluded[:, None]).abs().sum().item(), 0)

    def test_nir_validity_does_not_mask_main_or_color_labels(self):
        _, first = self.objective(self.outputs, self.target, True, auxiliary_valid={"nir": self.valid})
        changed_target = self.target.clone()
        excluded = (~self.valid[:, 0]) & (self.target != 255)
        changed_target[excluded] = 1 - changed_target[excluded]
        _, second = self.objective(self.outputs, changed_target, True, auxiliary_valid={"nir": self.valid})
        torch.testing.assert_close(first["aux_nir"], second["aux_nir"])
        for key in ("main", "aux_rgb", "aux_hsv", "aux_lab"):
            self.assertFalse(torch.isclose(first[key], second[key]).item(), key)

    def test_all_invalid_nir_is_finite_zero_with_zero_branch_gradients(self):
        # An empty support must not sum half-precision logits into an overflow.
        self.outputs["aux"]["nir"] = torch.full((2, 2, 2, 3), 60000.0,
                                                   dtype=torch.float16, requires_grad=True)
        total, parts = self.objective(self.outputs, self.target, True,
                                     auxiliary_valid={"nir": torch.zeros_like(self.valid)})
        self.assertTrue(torch.isfinite(total).item())
        self.assertEqual(parts["aux_nir"].item(), 0)
        total.backward()
        nir = self.outputs["aux"]["nir"]
        torch.testing.assert_close(nir.grad, torch.zeros_like(nir))
        self.assertGreater(self.outputs["aux"]["rgb"].grad.abs().sum().item(), 0)

    def test_all_ignored_targets_have_finite_zero_loss_and_gradients(self):
        total, parts = self.objective(self.outputs, torch.full_like(self.target, 255), True,
                                     auxiliary_valid={"nir": self.valid})
        self.assertEqual(total.item(), 0)
        self.assertTrue(all(torch.isfinite(value).item() for value in parts.values()))
        total.backward()
        for logits in (self.outputs["out"], *self.outputs["aux"].values()):
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_combo_minmax_ignore_is_shared_by_all_heads_and_intersected_with_nir_validity(self):
        target = self.target.clone()
        target[:, 1, 0] = 255
        objective = SegmentationObjective(FocalDiceLoss({"focal_weight": 2.0}, 255), 255, 0.4)
        _, parts = objective(self.outputs, target, True, auxiliary_valid={"nir": self.valid})
        for name in ("rgb", "hsv", "lab"):
            torch.testing.assert_close(parts["aux_" + name],
                                       F.cross_entropy(self.outputs["aux"][name], target, ignore_index=255))
        nir_target = target.masked_fill(~self.valid[:, 0], 255)
        torch.testing.assert_close(parts["aux_nir"],
                                   F.cross_entropy(self.outputs["aux"]["nir"], nir_target, ignore_index=255))
        parts["total"].backward()
        for name, logits in (("out", self.outputs["out"]), *self.outputs["aux"].items()):
            excluded = target.eq(255)[:, None]
            if name == "nir":
                excluded = excluded | ~self.valid
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad.masked_select(excluded).abs().sum().item(), 0)
            self.assertGreater(logits.grad.masked_select(~excluded).abs().sum().item(), 0)

    def test_combo_all_ignored_half_logits_have_finite_zero_main_and_four_auxiliary_losses(self):
        outputs = {"out": torch.full((2, 2, 2, 3), 60000.0, dtype=torch.float16, requires_grad=True),
                   "aux": {name: torch.full((2, 2, 2, 3), 60000.0, dtype=torch.float16,
                                            requires_grad=True)
                           for name in ("rgb", "hsv", "lab", "nir")}}
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        loss, parts = objective(outputs, torch.full_like(self.target, 255), True,
                                auxiliary_valid={"nir": self.valid})
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        self.assertTrue(all(value.item() == 0 for value in parts.values()))
        loss.backward()
        for logits in (outputs["out"], *outputs["aux"].values()):
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_nir_requires_explicit_boolean_mask_with_matching_shape_and_device(self):
        for validity in (None, {}, {"rgb": self.valid}):
            with self.subTest(validity=validity), self.assertRaisesRegex(ValueError, "auxiliary_valid"):
                self.objective(self.outputs, self.target, auxiliary_valid=validity)
        with self.assertRaisesRegex(TypeError, "boolean"):
            self.objective(self.outputs, self.target, auxiliary_valid={"nir": self.valid.float()})
        with self.assertRaisesRegex(ValueError, "shape"):
            self.objective(self.outputs, self.target, auxiliary_valid={"nir": self.valid[:, 0]})
        with self.assertRaisesRegex(ValueError, "device"):
            self.objective(self.outputs, self.target,
                           auxiliary_valid={"nir": torch.ones(self.valid.shape, dtype=torch.bool, device="meta")})

    def test_branch_sets_must_be_exactly_three_color_heads_or_four_with_nir(self):
        for names in (("nir",), ("rgb", "hsv", "nir"), ("rgb", "hsv", "lab", "other")):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "rgb"):
                outputs = {"out": self.outputs["out"],
                           "aux": {name: self.outputs["out"] for name in names}}
                self.objective(outputs, self.target, auxiliary_valid={"nir": self.valid})


class NirMixUpTests(_CpuTests):
    def setUp(self):
        self.images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        self.masks = torch.tensor([[[0, 1], [255, 0]], [[1, 0], [1, 255]]])
        self.nir = torch.tensor([[[[10.0, 200.0], [30.0, 400.0]]],
                                 [[[100.0, 20.0], [300.0, 40.0]]]], requires_grad=True)
        self.valid = torch.tensor([[[[True, False], [True, False]]],
                                   [[[True, True], [False, False]]]])

    def test_shared_pairing_and_lambda_use_only_valid_nir_support(self):
        originals = [value.detach().clone() for value in (self.images, self.masks, self.nir, self.valid)]
        permutation = torch.tensor([1, 0])
        with patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.25)), \
                patch("torch.randperm", return_value=permutation) as permute:
            batch = BatchMixUp(probability=1).with_nir(self.images, self.masks, self.nir, self.valid)
        self.assertEqual(set(batch), {"images", "target_a", "target_b", "lam", "nir", "nir_valid", "valid_a", "valid_b"})
        self.assertEqual(batch["lam"], 0.25)
        permute.assert_called_once()
        torch.testing.assert_close(batch["images"], 0.25 * self.images + 0.75 * self.images[permutation])
        torch.testing.assert_close(batch["target_a"], self.masks)
        torch.testing.assert_close(batch["target_b"], self.masks[permutation])
        torch.testing.assert_close(batch["valid_a"], self.valid)
        torch.testing.assert_close(batch["valid_b"], self.valid[permutation])
        expected = torch.tensor([[[[77.5, 20.0], [30.0, 0.0]]],
                                  [[[32.5, 20.0], [30.0, 0.0]]]])
        torch.testing.assert_close(batch["nir"], expected)
        torch.testing.assert_close(batch["nir_valid"], self.valid | self.valid[permutation])
        self.assertEqual(batch["nir_valid"].dtype, torch.bool)
        batch["nir"].sum().backward()
        self.assertEqual(self.nir.grad.masked_select(~self.valid).abs().sum().item(), 0)
        for original, current in zip(originals, (self.images, self.masks, self.nir, self.valid)):
            torch.testing.assert_close(current, original)

    def test_sampling_matches_old_mixup_for_same_seed(self):
        torch.manual_seed(37)
        old = BatchMixUp(probability=1)(self.images, self.masks)
        torch.manual_seed(37)
        new = BatchMixUp(probability=1).with_nir(self.images, self.masks, self.nir, self.valid)
        self.assertEqual(len(old), 4)
        for first, key in zip(old[:3], ("images", "target_a", "target_b")):
            torch.testing.assert_close(first, new[key], rtol=0, atol=0)
        self.assertEqual(old[3], new["lam"])

    def test_disabled_singleton_and_rejected_draw_keep_data_and_source_validity(self):
        for probability, count in ((0, 2), (1, 1), (0.5, 2)):
            images, masks, nir, valid = (value[:count] for value in
                                         (self.images, self.masks, self.nir, self.valid))
            with self.subTest(probability=probability, count=count), \
                    patch("torch.rand", return_value=torch.tensor(0.9)), \
                    patch("torch.distributions.Beta.sample") as sample, \
                    patch("torch.randperm") as permute:
                batch = BatchMixUp(probability=probability).with_nir(images, masks, nir, valid)
                for key, value in (("images", images), ("target_a", masks), ("target_b", masks),
                                   ("nir", nir), ("nir_valid", valid), ("valid_a", valid), ("valid_b", valid)):
                    self.assertIs(batch[key], value)
                self.assertEqual(batch["lam"], 1)
                sample.assert_not_called()
                permute.assert_not_called()

    def test_endpoint_lambdas_only_use_the_nonzero_weight_source(self):
        permutation = torch.tensor([1, 0])
        for lam in (0.0, 1.0):
            with self.subTest(lam=lam), \
                    patch("torch.distributions.Beta.sample", return_value=torch.tensor(lam)), \
                    patch("torch.randperm", return_value=permutation):
                batch = BatchMixUp(probability=1).with_nir(self.images, self.masks, self.nir, self.valid)
                source = self.nir if lam else self.nir[permutation]
                valid = self.valid if lam else self.valid[permutation]
                torch.testing.assert_close(batch["nir_valid"], valid)
                torch.testing.assert_close(batch["nir"], torch.where(valid, source, torch.zeros_like(source)))

    def test_rejects_incompatible_shapes_dtypes_devices_and_nonfinite_values(self):
        mixer = BatchMixUp(probability=0)
        with self.assertRaisesRegex(ValueError, "shapes"):
            mixer.with_nir(self.images, self.masks, self.nir[:, 0], self.valid)
        with self.assertRaisesRegex(TypeError, "boolean"):
            mixer.with_nir(self.images, self.masks, self.nir, self.valid.float())
        with self.assertRaisesRegex(TypeError, "floating"):
            mixer.with_nir(self.images, self.masks, self.nir.long(), self.valid)
        with self.assertRaisesRegex(ValueError, "device"):
            mixer.with_nir(self.images, self.masks, torch.empty(self.nir.shape, device="meta"), self.valid)
        for field in ("rgb", "nir"):
            images, nir = self.images.clone(), self.nir.detach().clone()
            (images if field == "rgb" else nir).reshape(-1)[0] = float("nan")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "finite"):
                mixer.with_nir(images, self.masks, nir, self.valid)


class NirMixUpObjectiveTests(_CpuTests):
    def test_combo_mixup_combines_per_source_minmax_ignore_and_nir_validity(self):
        torch.manual_seed(27)
        outputs = {"out": torch.randn(2, 2, 2, 3, requires_grad=True),
                   "aux": {name: torch.randn(2, 2, 2, 3, requires_grad=True)
                           for name in ("rgb", "hsv", "lab", "nir")}}
        target_a = torch.tensor([[[255, 0, 255], [1, 255, 1]],
                                 [[255, 1, 0], [255, 1, 0]]])
        target_b = torch.tensor([[[255, 255, 1], [0, 1, 255]],
                                 [[255, 0, 255], [1, 255, 1]]])
        valid_a = torch.ones(2, 1, 2, 3, dtype=torch.bool)
        valid_b = torch.ones_like(valid_a)
        valid_a[:, :, 1, 0] = False
        valid_b[:, :, 0, 2] = False
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        first, parts_a = objective(outputs, target_a, True, auxiliary_valid={"nir": valid_a})
        second, parts_b = objective(outputs, target_b, True, auxiliary_valid={"nir": valid_b})
        actual, parts = mixup_objective(objective, outputs, target_a, target_b, 0.25,
                                       auxiliary_valid_a={"nir": valid_a},
                                       auxiliary_valid_b={"nir": valid_b})
        torch.testing.assert_close(actual, 0.25 * first + 0.75 * second)
        for name in parts:
            torch.testing.assert_close(parts[name], 0.25 * parts_a[name] + 0.75 * parts_b[name])
        actual.backward()
        for name, logits in (("out", outputs["out"]), *outputs["aux"].items()):
            support_a, support_b = (target_a != 255)[:, None], (target_b != 255)[:, None]
            if name == "nir":
                support_a, support_b = support_a & valid_a, support_b & valid_b
            excluded = ~(support_a | support_b)
            self.assertEqual(logits.grad.masked_select(excluded).abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.masked_select(~excluded).abs().sum().item(), 0)

    def test_combo_nir_mixup_all_ignored_and_single_ignored_sources_are_finite(self):
        target = torch.tensor([[[0, 255, 1], [1, 0, 1]]])
        ignored = torch.full_like(target, 255)
        validity = {"nir": torch.ones(1, 1, 2, 3, dtype=torch.bool)}
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        for target_a, target_b, valid_weight in ((ignored, ignored, 0.0),
                                                 (ignored, target, 0.75), (target, ignored, 0.25)):
            with self.subTest(valid_weight=valid_weight):
                outputs = {"out": torch.randn(1, 2, 2, 3, requires_grad=True),
                           "aux": {name: torch.randn(1, 2, 2, 3, requires_grad=True)
                                   for name in ("rgb", "hsv", "lab", "nir")}}
                actual, parts = mixup_objective(objective, outputs, target_a, target_b, 0.25,
                                               auxiliary_valid_a=validity, auxiliary_valid_b=validity)
                expected = objective(outputs, target, auxiliary_valid=validity)
                torch.testing.assert_close(actual, valid_weight * expected)
                self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
                actual.backward()
                for logits in (outputs["out"], *outputs["aux"].values()):
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    if valid_weight == 0:
                        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
                    else:
                        self.assertEqual(logits.grad.masked_select(target.eq(255)[:, None]).abs().sum().item(), 0)
                        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_each_target_uses_its_own_nir_mask_and_components_are_weighted(self):
        torch.manual_seed(11)
        outputs = {"out": torch.randn(2, 2, 2, 3, requires_grad=True),
                   "aux": {name: torch.randn(2, 2, 2, 3, requires_grad=True)
                           for name in ("rgb", "hsv", "lab", "nir")}}
        target_a = torch.randint(0, 2, (2, 2, 3))
        target_b = 1 - target_a
        valid_a = torch.zeros(2, 1, 2, 3, dtype=torch.bool)
        valid_b = torch.zeros_like(valid_a)
        valid_a[:, :, 0, 0] = True
        valid_b[:, :, 1, 1] = True
        objective = SegmentationObjective(nn.CrossEntropyLoss(ignore_index=255), 255, 0.4)
        first, parts_a = objective(outputs, target_a, True, auxiliary_valid={"nir": valid_a})
        second, parts_b = objective(outputs, target_b, True, auxiliary_valid={"nir": valid_b})
        loss, parts = mixup_objective(objective, outputs, target_a, target_b, 0.25,
                                     auxiliary_valid_a={"nir": valid_a}, auxiliary_valid_b={"nir": valid_b})
        torch.testing.assert_close(loss, 0.25 * first + 0.75 * second)
        for name in parts:
            torch.testing.assert_close(parts[name], 0.25 * parts_a[name] + 0.75 * parts_b[name])
        loss.backward()
        excluded = ~(valid_a | valid_b)
        self.assertEqual(outputs["aux"]["nir"].grad.masked_select(excluded).abs().sum().item(), 0)

    def test_endpoint_lambda_only_passes_selected_target_validity(self):
        outputs, target_a, target_b = object(), torch.tensor([0]), torch.tensor([1])
        valid_a, valid_b = {"nir": torch.tensor([True])}, {"nir": torch.tensor([False])}
        for lam, selected, validity in ((1, target_a, valid_a), (0, target_b, valid_b)):
            calls = []

            def criterion(logits, target, return_components=False, *, auxiliary_valid=None):
                calls.append((target, auxiliary_valid))
                result = target.float().sum()
                return (result, {"total": result}) if return_components else result

            result = mixup_objective(criterion, outputs, target_a, target_b, lam,
                                     auxiliary_valid_a=valid_a, auxiliary_valid_b=valid_b,
                                     return_components=False)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], selected)
            self.assertIs(calls[0][1], validity)
            torch.testing.assert_close(result, selected.float().sum())

    def test_old_criterion_is_not_given_an_unknown_validity_keyword(self):
        def old_criterion(outputs, target, return_components=False):
            loss = target.float().mean()
            return (loss, {"total": loss}) if return_components else loss

        for lam in (0, 0.25, 1):
            result, parts = mixup_objective(old_criterion, None, torch.tensor([2]), torch.tensor([6]), lam)
            torch.testing.assert_close(result, torch.tensor(lam * 2 + (1 - lam) * 6, dtype=torch.float32))
            torch.testing.assert_close(parts["total"], result)


if __name__ == "__main__":
    unittest.main()

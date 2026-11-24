"""Checks for branch gradients and ignored native-crop padding."""

import unittest

import torch
from torch import nn

from losses import FocalDiceLoss, SegmentationObjective, build_criterion


class SegmentationObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(13)

    def test_auxiliary_heads_receive_gradients_independent_of_final_output(self):
        outputs = {
            "out": torch.randn(2, 2, 4, 5, requires_grad=True),
            "aux": {name: torch.randn(2, 2, 4, 5, requires_grad=True)
                    for name in ("rgb", "hsv", "lab")},
        }
        target = torch.randint(0, 2, (2, 4, 5))
        objective = SegmentationObjective(nn.CrossEntropyLoss(ignore_index=255), 255, 0.4)
        total, parts = objective(outputs, target, return_components=True)
        expected_aux = torch.stack([parts["aux_" + n] for n in ("rgb", "hsv", "lab")]).mean()
        torch.testing.assert_close(total, parts["main"] + 0.4 * expected_aux)
        total.backward()
        for logits in [outputs["out"], *outputs["aux"].values()]:
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_padding_never_changes_main_or_auxiliary_loss(self):
        target = torch.tensor([[[0, 1, 255], [1, 0, 255]]])
        first = torch.randn(1, 2, 2, 3)
        second = first.clone()
        second[:, :, :, 2] = torch.tensor([[-100.0, 100.0]]).reshape(1, 2, 1)
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        result = []
        for logits in (first, second):
            result.append(objective({"out": logits, "aux": {n: logits for n in ("rgb", "hsv", "lab")}}, target))
        torch.testing.assert_close(result[0], result[1])

    def test_all_ignored_crop_has_finite_zero_loss_and_gradients(self):
        logits = torch.randn(1, 2, 3, 4, requires_grad=True)
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        target = torch.full((1, 3, 4), 255, dtype=torch.long)
        loss = objective({"out": logits, "aux": {n: logits for n in ("rgb", "hsv", "lab")}}, target)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_combo_itself_handles_all_ignore_without_empty_mean_or_half_overflow(self):
        target = torch.full((2, 4, 5), 255, dtype=torch.long)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                logits = torch.full((2, 2, 4, 5), 60000.0, dtype=dtype, requires_grad=True)
                loss = FocalDiceLoss({"focal_weight": 2.0, "focal_gamma": 3.0}, 255)(logits, target)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_minmax_ignore_regions_have_zero_main_and_color_auxiliary_gradients(self):
        target = torch.tensor([[[0, 255, 1], [255, 1, 0]]])
        original_target = target.clone()
        excluded = target.eq(255)[:, None]
        outputs = {"out": torch.randn(1, 2, 2, 3, requires_grad=True),
                   "aux": {name: torch.randn(1, 2, 2, 3, requires_grad=True)
                           for name in ("rgb", "hsv", "lab")}}
        altered = {"out": outputs["out"].detach().masked_fill(excluded, 60000),
                   "aux": {name: logits.detach().masked_fill(excluded, -60000)
                           for name, logits in outputs["aux"].items()}}
        objective = SegmentationObjective(FocalDiceLoss({"focal_gamma": 3.0}, 255), 255, 0.4)
        loss, parts = objective(outputs, target, return_components=True)
        changed_loss, changed_parts = objective(altered, target, return_components=True)
        torch.testing.assert_close(loss, changed_loss)
        for name in parts:
            torch.testing.assert_close(parts[name], changed_parts[name])
        loss.backward()
        for logits in (outputs["out"], *outputs["aux"].values()):
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad.masked_select(excluded).abs().sum().item(), 0)
            self.assertGreater(logits.grad.masked_select(~excluded).abs().sum().item(), 0)
        torch.testing.assert_close(target, original_target)

    def test_adding_fully_ignored_samples_does_not_dilute_valid_combo_or_auxiliary_losses(self):
        target = torch.tensor([[[0, 255, 1], [1, 0, 1]]])
        outputs = {"out": torch.randn(1, 2, 2, 3),
                   "aux": {name: torch.randn(1, 2, 2, 3) for name in ("rgb", "hsv", "lab")}}

        def append_ignored(logits):
            return torch.cat((logits, torch.full_like(logits, 60000)), dim=0)

        padded = {"out": append_ignored(outputs["out"]),
                  "aux": {name: append_ignored(logits) for name, logits in outputs["aux"].items()}}
        padded_target = torch.cat((target, torch.full_like(target, 255)), dim=0)
        objective = SegmentationObjective(FocalDiceLoss({}, 255), 255, 0.4)
        _, first = objective(outputs, target, return_components=True)
        _, second = objective(padded, padded_target, return_components=True)
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_auxiliary_configuration_cannot_silently_skip_heads(self):
        config = {"model": {"variant": "baseline"}, "loss": {"auxiliary_weight": 0.4}}
        with self.assertRaisesRegex(ValueError, "auxiliary_supervision"):
            build_criterion(config, "cpu")


if __name__ == "__main__":
    unittest.main()

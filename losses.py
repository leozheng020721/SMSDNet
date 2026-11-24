"""Final-output segmentation losses with optional color/NIR supervision."""

import torch
from torch import nn


class FocalDiceLoss(nn.Module):
    def __init__(self, options, ignore_index):
        super().__init__()
        import segmentation_models_pytorch as smp

        alpha = options.get("focal_alpha", 0.25)
        if isinstance(alpha, (list, tuple)):
            raise ValueError("focal_alpha must be a scalar, not class weights")
        self.focal = smp.losses.FocalLoss(
            mode="multiclass", alpha=alpha,
            gamma=options.get("focal_gamma", 2.0), ignore_index=ignore_index,
        )
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=ignore_index,
        )
        self.focal_weight = float(options.get("focal_weight", 1.0))
        self.dice_weight = float(options.get("dice_weight", 1.0))
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        if self.ignore_index is not None and not (target != self.ignore_index).any():
            # SMP focal averages an empty tensor when every label is ignored.
            return logits.reshape(-1)[:0].sum()
        # Keep ignore labels intact so padding never becomes background in Dice.
        return (
            self.focal_weight * self.focal(logits, target)
            + self.dice_weight * self.dice(logits, target)
        )


class SegmentationObjective(nn.Module):
    def __init__(self, main_loss, ignore_index=255, auxiliary_weight=0.0):
        super().__init__()
        if auxiliary_weight < 0:
            raise ValueError("auxiliary_weight cannot be negative")
        self.main_loss = main_loss
        self.ignore_index = ignore_index
        self.auxiliary_weight = float(auxiliary_weight)
        self.auxiliary_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs, target, return_components=False, *, auxiliary_valid=None):
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        has_valid_pixels = (target != self.ignore_index).any()
        main = self.main_loss(logits, target) if has_valid_pixels else logits.reshape(-1)[:0].sum()
        auxiliary = main.new_zeros(())
        components = {"main": main}

        # Tensor-only input is also used for main-only sliding-window validation.
        if isinstance(outputs, dict) and self.auxiliary_weight > 0:
            branches = outputs.get("aux", {})
            names = set(branches)
            if names not in ({"rgb", "hsv", "lab"}, {"rgb", "hsv", "lab", "nir"}):
                raise ValueError("Auxiliary supervision requires rgb, hsv and lab logits, optionally nir")
            nir_valid = None
            if "nir" in names:
                if not isinstance(auxiliary_valid, dict) or "nir" not in auxiliary_valid:
                    raise ValueError("NIR auxiliary supervision requires auxiliary_valid['nir']")
                nir_valid = auxiliary_valid["nir"]
                if not isinstance(nir_valid, torch.Tensor) or nir_valid.dtype != torch.bool:
                    raise TypeError("NIR auxiliary validity must be a boolean tensor")
                if nir_valid.shape != (target.shape[0], 1, *target.shape[-2:]):
                    raise ValueError("NIR auxiliary validity must have shape Bx1xHxW matching the target")
                if nir_valid.device != target.device or nir_valid.device != branches["nir"].device:
                    raise ValueError("NIR auxiliary validity, logits and target must share the same device")
            branch_losses = []
            for name in ("rgb", "hsv", "lab") + (("nir",) if "nir" in names else ()):
                branch = branches[name]
                if branch.dtype in (torch.float16, torch.bfloat16):
                    branch = branch.float()
                if name == "nir":
                    nir_target = target.masked_fill(~nir_valid[:, 0], self.ignore_index)
                    value = (self.auxiliary_loss(branch, nir_target)
                             if (nir_target != self.ignore_index).any()
                             else branch.reshape(-1)[:0].sum())
                else:
                    value = (self.auxiliary_loss(branch, target)
                             if has_valid_pixels else branch.reshape(-1)[:0].sum())
                components["aux_" + name] = value
                branch_losses.append(value)
            auxiliary = torch.stack(branch_losses).mean()

        total = main + self.auxiliary_weight * auxiliary
        components.update(aux=auxiliary, total=total)
        return (total, components) if return_components else total


def build_criterion(config, device):
    options = config.get("loss", {})
    ignore_index = config["model"].get("ignore_index", 255)
    loss_type = options.get("type", "ce")
    auxiliary_weight = float(options.get("auxiliary_weight", 0.0))
    if auxiliary_weight > 0 and not (
        config["model"].get("variant") == "multicolor"
        and config["model"].get("multicolor", {}).get("auxiliary_supervision", False)
    ):
        raise ValueError("auxiliary_weight requires a multicolor model with auxiliary_supervision")

    if loss_type == "combo":
        main_loss = FocalDiceLoss(options, ignore_index)
    elif loss_type == "focal":
        import segmentation_models_pytorch as smp
        main_loss = smp.losses.FocalLoss(
            mode="multiclass", alpha=options.get("focal_alpha", 0.25),
            gamma=options.get("focal_gamma", 2.0), ignore_index=ignore_index,
        )
    elif loss_type in ("ce", "weighted_ce"):
        weights = options.get("class_weights") if loss_type == "weighted_ce" else None
        weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        main_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return SegmentationObjective(main_loss, ignore_index, auxiliary_weight).to(device)

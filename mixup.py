"""Batch MixUp with weighted hard-target segmentation losses.

Focal/Dice losses are evaluated separately against each hard target. Their
weighted sum is not algebraically equivalent to Dice on a soft mixed mask.
"""

import math

import torch


class BatchMixUp:
    def __init__(self, alpha=0.2, probability=0.5):
        self.alpha = float(alpha)
        self.probability = float(probability)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("MixUp alpha must be finite and positive")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("MixUp probability must be between zero and one")

    @staticmethod
    def _validate_images_masks(images, masks):
        if images.ndim != 4 or masks.ndim != 3:
            raise ValueError("MixUp expects BxCxHxW images and BxHxW masks")
        if images.shape[0] != masks.shape[0] or images.shape[-2:] != masks.shape[-2:]:
            raise ValueError("MixUp image/mask batch and spatial sizes must match")
        if images.device != masks.device:
            raise ValueError("MixUp images and masks must share the same device")
        if not images.is_floating_point():
            raise TypeError("MixUp images must be floating point")
        if masks.dtype != torch.long:
            raise TypeError("MixUp masks must retain integer long labels")

    def _sample(self, images):
        if self.probability == 0 or images.shape[0] < 2:
            return 1.0, None
        if self.probability < 1 and torch.rand((), device=images.device).item() >= self.probability:
            return 1.0, None

        concentration = torch.tensor(self.alpha, dtype=torch.float32, device=images.device)
        lam = float(torch.distributions.Beta(concentration, concentration).sample().item())
        permutation = torch.randperm(images.shape[0], device=images.device)
        return lam, permutation

    def __call__(self, images, masks):
        self._validate_images_masks(images, masks)
        lam, permutation = self._sample(images)
        if permutation is None:
            return images, masks, masks, lam
        mixed_images = lam * images + (1.0 - lam) * images[permutation]
        return mixed_images, masks, masks[permutation], lam

    def with_nir(self, images, masks, nir, nir_valid):
        """Mix aligned modalities while preserving each hard target's support."""
        self._validate_images_masks(images, masks)
        expected_shape = (images.shape[0], 1, *images.shape[-2:])
        if not isinstance(nir, torch.Tensor) or not nir.is_floating_point():
            raise TypeError("MixUp NIR must be a floating-point tensor")
        if not isinstance(nir_valid, torch.Tensor) or nir_valid.dtype != torch.bool:
            raise TypeError("MixUp NIR validity must be a boolean tensor")
        if nir.shape != expected_shape or nir_valid.shape != expected_shape:
            raise ValueError("MixUp NIR and validity must have matching Bx1xHxW shapes")
        if nir.device != images.device or nir_valid.device != images.device:
            raise ValueError("MixUp RGB, NIR and validity must share the same device")
        if not torch.isfinite(images).all() or not torch.isfinite(nir).all():
            raise ValueError("MixUp RGB and NIR inputs must be finite")
        lam, permutation = self._sample(images)
        if permutation is None:
            return {"images": images, "target_a": masks, "target_b": masks, "lam": lam,
                    "nir": nir, "nir_valid": nir_valid, "valid_a": nir_valid, "valid_b": nir_valid}

        valid_a, valid_b = nir_valid, nir_valid[permutation]
        # Accumulate support in at least float32, including under mixed precision.
        work_dtype = torch.float64 if nir.dtype == torch.float64 else torch.float32
        weight_a = lam * valid_a.to(work_dtype)
        weight_b = (1.0 - lam) * valid_b.to(work_dtype)
        support = weight_a + weight_b
        present = support > 0
        source = torch.where(nir_valid, nir, torch.zeros_like(nir)).to(work_dtype)
        numerator = weight_a * source + weight_b * source[permutation]
        denominator = torch.where(present, support, torch.ones_like(support))
        mixed_nir = (numerator / denominator).to(nir.dtype)
        return {"images": lam * images + (1.0 - lam) * images[permutation],
                "target_a": masks, "target_b": masks[permutation], "lam": lam,
                "nir": mixed_nir, "nir_valid": present, "valid_a": valid_a, "valid_b": valid_b}


def mixup_objective(criterion, outputs, target_a, target_b, lam, return_components=True, *,
                    auxiliary_valid_a=None, auxiliary_valid_b=None):
    """Weight main/auxiliary losses for two targets without another forward."""
    lam = float(lam)
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("MixUp lambda must be between zero and one")
    kwargs_a = {} if auxiliary_valid_a is None else {"auxiliary_valid": auxiliary_valid_a}
    kwargs_b = {} if auxiliary_valid_b is None else {"auxiliary_valid": auxiliary_valid_b}
    if lam == 1:
        return criterion(outputs, target_a, return_components=return_components, **kwargs_a)
    if lam == 0:
        return criterion(outputs, target_b, return_components=return_components, **kwargs_b)

    loss_a, parts_a = criterion(outputs, target_a, return_components=True, **kwargs_a)
    loss_b, parts_b = criterion(outputs, target_b, return_components=True, **kwargs_b)
    if parts_a.keys() != parts_b.keys():
        raise ValueError("MixUp requires matching loss components for both targets")
    loss = lam * loss_a + (1.0 - lam) * loss_b
    parts = {name: lam * value + (1.0 - lam) * parts_b[name]
             for name, value in parts_a.items()}
    return (loss, parts) if return_components else loss

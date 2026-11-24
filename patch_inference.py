"""Native-resolution tiling without transferring whole images to the GPU."""

import torch
from torch.nn import functional as F


def is_crop_mode(config):
    mode = config['dataset'].get('spatial_mode', 'resize').lower()
    if mode not in {'resize', 'crop'}:
        raise ValueError("dataset.spatial_mode must be 'resize' or 'crop'")
    return mode == 'crop'


def _tile_starts(length, tile_size, stride):
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def sliding_window_logits(model, images, device, tile_size=512, stride=384, tile_batch_size=2,
                          nir=None, nir_valid=None):
    """Average overlapping tile logits into one original-size CPU prediction.

    The caller sets ``model.eval()`` and supplies normalized BxCxHxW CPU
    tensors. Only tile batches are transferred to ``device``. Each output
    pixel is accumulated on CPU and divided by its own coverage count.
    """
    if images.ndim != 4 or images.shape[0] != 1:
        raise ValueError("sliding_window_logits requires one BxCxHxW image (B=1)")
    if images.device.type != 'cpu':
        raise ValueError("Keep full-resolution images on CPU for sliding-window inference")
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(stride, int) or not 0 < stride <= tile_size:
        raise ValueError("stride must satisfy 0 < stride <= tile_size")
    if not isinstance(tile_batch_size, int) or tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be a positive integer")
    height, width = images.shape[-2:]
    if height == 0 or width == 0:
        raise ValueError("Image height and width must be positive")
    padding = (0, max(0, tile_size - width), 0, max(0, tile_size - height))
    padded = F.pad(images, padding)
    padded_modalities = {}
    if (nir is None) != (nir_valid is None):
        raise ValueError("Sliding-window NIR requires both nir and nir_valid")
    if nir is not None:
        for name, value in (('nir', nir), ('nir_valid', nir_valid)):
            if tuple(value.shape) != (1, 1, height, width) or value.device.type != 'cpu':
                raise ValueError(f"Keep {name} as an aligned 1x1xHxW CPU tensor")
            padded_modalities[name] = F.pad(value, padding, value=0)
    padded_height, padded_width = padded.shape[-2:]
    positions = [
        (top, left)
        for top in _tile_starts(padded_height, tile_size, stride)
        for left in _tile_starts(padded_width, tile_size, stride)
    ]
    accumulated = None
    coverage = torch.zeros((1, 1, padded_height, padded_width), dtype=torch.float32)
    for offset in range(0, len(positions), tile_batch_size):
        batch_positions = positions[offset:offset + tile_batch_size]
        tiles = torch.cat([
            padded[:, :, top:top + tile_size, left:left + tile_size]
            for top, left in batch_positions
        ], dim=0).to(device)
        modalities = {
            name: torch.cat([
                value[:, :, top:top + tile_size, left:left + tile_size]
                for top, left in batch_positions
            ], dim=0).to(device)
            for name, value in padded_modalities.items()
        }
        logits = model(tiles, **modalities)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise ValueError("Model must return BxClassesxHxW main logits")
        if logits.shape[0] != len(batch_positions) or logits.shape[-2:] != (tile_size, tile_size):
            raise ValueError("Model logits must preserve each tile's batch and spatial sizes")
        logits = logits.detach().to(device='cpu', dtype=torch.float32)
        if accumulated is None:
            accumulated = torch.zeros((1, logits.shape[1], padded_height, padded_width), dtype=torch.float32)
        elif logits.shape[1] != accumulated.shape[1]:
            raise ValueError("Model returned inconsistent class counts across tile batches")
        for index, (top, left) in enumerate(batch_positions):
            accumulated[:, :, top:top + tile_size, left:left + tile_size] += logits[index:index + 1]
            coverage[:, :, top:top + tile_size, left:left + tile_size] += 1
    if not torch.all(coverage > 0):
        raise RuntimeError("Sliding-window positions did not cover every image pixel")
    return (accumulated / coverage)[:, :, :height, :width]

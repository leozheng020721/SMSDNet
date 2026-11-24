"""Shared batch handling for RGB and optional aligned NIR inputs."""


def unpack_segmentation_batch(batch):
    if isinstance(batch, dict):
        if not {"image", "mask"}.issubset(batch):
            raise ValueError("Segmentation batches require image and mask")
        modalities = {name: batch[name] for name in ("nir", "nir_valid") if name in batch}
        if modalities and set(modalities) != {"nir", "nir_valid"}:
            raise ValueError("NIR batches require both nir and nir_valid")
        return batch["image"], batch["mask"], modalities
    images, masks = batch
    return images, masks, {}

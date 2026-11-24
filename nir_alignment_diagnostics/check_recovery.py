"""Inspect the first three latest rgb_ready products without refitting anything."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio

from align_seo_nir import stretch_rgb, warp_nir


ROOT = Path(r"E:\Dataset\SEO")
OUTPUT = Path(__file__).parent / "recovery_checks"


def resize(image):
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return Image.fromarray(image).resize((400, 420))


def main():
    cv2.setNumThreads(1)
    latest = {}
    with (ROOT / "nir_aligned_v1" / "pan_recovery_registration.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[row["key"]] = row
    selected = [row for row in latest.values() if row.get("status") == "rgb_ready"][:3]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for row in selected:
        aoi, stem = row["key"].split("/")
        with rasterio.open(ROOT / "msi" / aoi / (stem + "_msi.tif")) as dataset:
            source = dataset.read()
            source_valid = dataset.read_masks(4) > 0
        source_rgb = stretch_rgb(np.moveaxis(source[[2, 1, 0]], 0, -1))
        with rasterio.open(ROOT / "crops" / aoi / (stem + "_pan.tif")) as dataset:
            pan = dataset.read(1).astype(np.float32)
        target = np.asarray(Image.open(ROOT / "pansharpened_he" / aoi /
                                      (stem + "_rgb.png")).convert("RGB"))
        with rasterio.open(row["product"]["nir"]) as dataset:
            nir = dataset.read(1)
            tags = dataset.tags()
        valid = np.asarray(Image.open(row["product"]["valid"])) > 0
        size = (target.shape[1], target.shape[0])
        h = np.asarray(row["H_native_to_target"], np.float64)
        expected, expected_valid = warp_nir(source[3], source_valid, h, size)
        aligned = cv2.warpPerspective(source_rgb, h, size, flags=cv2.INTER_LINEAR)
        edges_target = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 50, 120)
        edges_source = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), 50, 120)
        overlay = np.stack((edges_source, edges_target, edges_target), axis=-1)
        lo, hi = np.percentile(pan, (1, 99))
        pan = np.uint8(np.clip((pan - lo) / max(hi - lo, 1), 0, 1) * 255)
        lo, hi = np.percentile(nir[valid], (1, 99))
        displayed_nir = np.uint8(np.clip((np.nan_to_num(nir, nan=lo) - lo)
                                         / max(hi - lo, 1), 0, 1) * 255)
        displayed_nir[~valid] = 0
        images = (target, pan, aligned, overlay, displayed_nir, np.uint8(valid) * 255)
        labels = ("Current RGB", "Original PAN", "Aligned MSI visible", "RGB cyan / MSI red",
                  "NIR display stretch", "Valid mask")
        canvas = Image.new("RGB", (2400, 450), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(resize(image), (400 * index, 30))
            draw.text((400 * index + 5, 8), label, fill="black")
        preview = OUTPUT / (stem + ".png")
        canvas.save(preview)
        mask_path = (ROOT / "shadows_max" / "shadow_max_compressed" / "shadow_masks" /
                     aoi / (stem + "_shadow.png"))
        shadow_size = Image.open(mask_path).size if mask_path.exists() else None
        result = {"key": row["key"], "preview": str(preview),
                  "finite_exactly_matches_valid": bool(np.array_equal(np.isfinite(nir), valid)),
                  "saved_nir_exactly_matches_recorded_warp": bool(np.array_equal(nir, expected, equal_nan=True)),
                  "saved_valid_exactly_matches_recorded_warp": bool(np.array_equal(valid, expected_valid)),
                  "H_tag_matches_record": bool(np.array_equal(h, np.asarray(json.loads(tags["H_NATIVE_TO_TARGET"])))),
                  "rgb_size_wh": size, "shadow_size_wh": shadow_size,
                  "valid_fraction": float(valid.mean())}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

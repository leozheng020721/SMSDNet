"""Draw shadow boundaries on RGB imagery and export lossless fixed-size tiles."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(r"E:\Dataset\21century\registration_output\rectified")
MASK_ROOT = ROOT / "test/results_century_nir_disabled"
OUTPUT_ROOT = ROOT / "test/shadow_outline_tiles_nonir"


def shadow_outline(shadow, valid, width):
    """Draw an inner boundary, excluding edges against unknown pixels."""
    kernel = np.ones((2 * width + 1, 2 * width + 1), dtype=np.uint8)
    foreground = ((shadow == 255) & valid).astype(np.uint8)
    interior = cv2.erode(foreground, kernel, borderType=cv2.BORDER_CONSTANT,
                         borderValue=0)
    known_neighborhood = cv2.erode(valid.astype(np.uint8), kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (foreground > 0) & (interior == 0) & (known_neighborhood > 0)


def check_grid(source, other, label):
    if (source.width, source.height, source.transform, source.crs) != (
            other.width, other.height, other.transform, other.crs):
        raise ValueError(f"{label} does not use the original image grid")
    if other.count != 1:
        raise ValueError(f"{label} must contain exactly one band")


def export_image(source_path, mask_directory, destination, args):
    stem = source_path.name.removesuffix("_rectified.tif")
    mask_path = mask_directory / f"{stem}_shadow_mask.tif"
    valid_path = mask_directory / f"{stem}_valid_mask.tif"
    input_paths = (source_path, mask_path, valid_path)
    original_stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths]
    records = []
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        masks = stack.enter_context(rasterio.open(mask_path))
        validity = stack.enter_context(rasterio.open(valid_path))
        check_grid(source, masks, "Shadow mask")
        check_grid(source, validity, "Validity mask")
        if source.descriptions != ("BLUE", "GREEN", "RED", "NIR"):
            raise ValueError(f"Unexpected source bands: {source.descriptions}")
        height, width = source.height, source.width
        halo = args.line_width
        for row, top in enumerate(range(0, height, args.tile_size), 1):
            for col, left in enumerate(range(0, width, args.tile_size), 1):
                actual_width = min(args.tile_size, width - left)
                actual_height = min(args.tile_size, height - top)
                # The halo makes morphology independent of the export grid.
                expanded = Window(left - halo, top - halo,
                                  args.tile_size + 2 * halo,
                                  args.tile_size + 2 * halo)
                binary = masks.read(1, window=expanded, boundless=True, fill_value=0)
                valid_values = validity.read(1, window=expanded, boundless=True,
                                            fill_value=0)
                if not np.isin(binary, (0, 255)).all() or not np.isin(valid_values, (0, 255)).all():
                    raise ValueError("Expected binary mask values 0 and 255")
                valid = valid_values == 255
                boundary = shadow_outline(binary, valid, args.line_width)
                center = (slice(halo, halo + args.tile_size),) * 2
                boundary = boundary[center]
                valid = valid[center]
                shadow = (binary[center] == 255) & valid
                window = Window(left, top, args.tile_size, args.tile_size)
                data = source.read([3, 2, 1], window=window, boundless=True,
                                   fill_value=0).astype(np.float32)
                rgb = np.moveaxis((np.clip(data / args.dn_max, 0, 1) * 255)
                                  .astype(np.uint8), 0, -1)
                rgb[~valid] = 0
                rgb[boundary] = (255, 0, 0)
                filename = f"{stem}_r{row:03d}_c{col:03d}_x{left:05d}_y{top:05d}.png"
                path = destination / filename
                Image.fromarray(rgb).save(path, compress_level=3)
                # Decode the written artifact, including padded edge tiles.
                with Image.open(path) as saved:
                    if saved.size != (args.tile_size, args.tile_size):
                        raise ValueError(f"Incorrect saved dimensions: {path}")
                    if not np.array_equal(np.asarray(saved), rgb):
                        raise ValueError(f"Lossless PNG verification failed: {path}")
                records.append({"file": str(path.resolve()), "row": row, "column": col,
                                "x": left, "y": top, "source_width": actual_width,
                                "source_height": actual_height,
                                "valid_pixels": int(valid.sum()),
                                "shadow_pixels": int(shadow.sum()),
                                "outline_pixels": int(boundary.sum())})
            print(f"  {stem}: row {row}/{(height + args.tile_size - 1) // args.tile_size}", flush=True)
    final_stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths]
    if original_stats != final_stats:
        raise ValueError("An input changed during export")
    return {"source": str(source_path), "mask": str(mask_path), "valid_mask": str(valid_path),
            "width": width, "height": height, "tile_count": len(records),
            "input_files_unchanged": True, "lossless_png_check_passed": True,
            "tiles": records}


def make_contact_sheet(images, destination):
    selected = []
    for record in images:
        candidates = sorted(record["tiles"], key=lambda tile: tile["outline_pixels"], reverse=True)
        selected.extend(candidates[:2])
    columns, cell = 2, 532
    canvas = Image.new("RGB", (columns * cell, ((len(selected) + 1) // 2) * (cell + 30)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, tile in enumerate(selected):
        x, y = (index % columns) * cell, (index // columns) * (cell + 30)
        with Image.open(tile["file"]) as original:
            preview = original.copy()
        preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x + 10, y + 30))
        draw.text((x + 10, y + 8), Path(tile["file"]).name, fill="black")
    canvas.save(destination, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--group", default="4")
    parser.add_argument("--region", default="region_01")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--dn-max", type=float, default=4095)
    args = parser.parse_args()
    if args.tile_size < 1 or args.line_width < 1 or not np.isfinite(args.dn_max) or args.dn_max <= 0:
        parser.error("Tile size, line width, and DN maximum must be positive")
    sources = sorted((args.input_root / args.group / args.region).glob("*_rectified.tif"))
    if not sources:
        parser.error("No rectified TIFF images found")
    destination = args.output / args.group / args.region
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"group": args.group, "region": args.region,
                "tile_size": args.tile_size, "overlap": 0, "edge_padding": "black",
                "rgb_bands": [3, 2, 1], "rgb_dn_max": args.dn_max,
                "outline_color_rgb": [255, 0, 0], "inner_outline_width": args.line_width,
                "invalid_boundary": "not outlined", "images": []}
    with rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024):
        for number, source_path in enumerate(sources, 1):
            stem = source_path.name.removesuffix("_rectified.tif")
            print(f"[{number}/{len(sources)}] {source_path.name}", flush=True)
            result = export_image(source_path, args.mask_root / args.group / args.region,
                                  destination / stem, args)
            manifest["images"].append(result)
    manifest["tile_count"] = sum(image["tile_count"] for image in manifest["images"])
    manifest["verification"] = "All saved PNGs decoded and compared exactly; input byte size and mtime unchanged"
    (destination / "tile_index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(manifest["images"], destination / "outline_samples.jpg")
    print(f"Finished {manifest['tile_count']} tiles: {destination}", flush=True)


if __name__ == "__main__":
    main()

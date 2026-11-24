"""Render native source/PAN/current RGB and report black padding and metadata."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compare_seo_nir_registration import load_pair, pan_display


ROOT = Path(r"E:\Dataset\SEO")
KEYS = ("JAX_436/JAX_436_8", "OMA_930/OMA_930_26", "UCSD_291/UCSD_291_3", "JAX_461/JAX_461_11")


def main():
    output = HERE / "input_review"
    output.mkdir(exist_ok=True)
    records = []
    for key in KEYS:
        visible, pan, rgb, valid = load_pair(ROOT, key)
        relative = Path(key)
        row = {"key": key, "rgb_nonblack_fraction": float((rgb.max(axis=2) > 2).mean()),
               "msi_valid_fraction": float(valid.mean()), "metadata": {}}
        for folder, suffix in (("msi", "_msi.tif"), ("crops", "_pan.tif")):
            with rasterio.open(ROOT / folder / relative.parent / (relative.name + suffix)) as dataset:
                row["metadata"][folder] = {"tag_keys": sorted(dataset.tags()),
                                            "rpc_present": bool(dataset.tags(ns="RPC")),
                                            "crs": str(dataset.crs), "size_wh": [dataset.width, dataset.height]}
        canvas = Image.new("RGB", (1200, 440), "white")
        draw = ImageDraw.Draw(canvas)
        panels = (("Native MSI visible", visible), ("Native PAN", pan_display(pan)), ("Current RGB", rgb))
        for index, (label, array) in enumerate(panels):
            panel = Image.fromarray(array).convert("RGB")
            panel.thumbnail((400, 400))
            canvas.paste(panel, (index * 400, 40))
            draw.text((index * 400 + 5, 10), label, fill="black")
        filename = output / (key.replace("/", "__") + ".jpg")
        canvas.save(filename, quality=95)
        row["preview"] = str(filename)
        records.append(row)
        print(json.dumps(row), flush=True)
    (output / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Read four difficult S-EO pairs; write diagnostics only in this directory."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio

from align_seo_nir import Options, _fit_and_check, stretch_rgb, warp_nir


ROOT = Path(r"E:\Dataset\SEO")
OUTPUT = Path(__file__).parent
EXAMPLES = ("JAX_307_8", "JAX_356_8", "JAX_486_15", "JAX_548_9")


def describe(array):
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "percentiles": np.percentile(array, [1, 10, 50, 90, 99]).tolist(),
        "nonzero_fraction": float(np.mean(array != 0)),
    }


def show_panel(array, width=400):
    if array.ndim == 2:
        lo, hi = np.percentile(array, (1, 99))
        array = np.uint8(np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
        array = np.repeat(array[:, :, None], 3, axis=2)
    height = round(array.shape[0] * width / array.shape[1])
    return Image.fromarray(array).resize((width, height))


def fit(source_gray, target_gray, working_size, target_size, threshold, ratio):
    sift = cv2.SIFT_create(nfeatures=15000, contrastThreshold=threshold, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(source_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    row = {"keypoints": [len(kp1), len(kp2)], "threshold": threshold, "ratio": ratio}
    if des1 is None or des2 is None or min(len(des1), len(des2)) < 2:
        row["error"] = "insufficient_descriptors"
        return row
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = [p[0] for p in matcher.knnMatch(des1, des2, k=2)
               if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    reverse = {(p[0].trainIdx, p[0].queryIdx)
               for p in matcher.knnMatch(des2, des1, k=2)
               if len(p) == 2 and p[0].distance < ratio * p[1].distance}
    matched = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
    row["mutual_matches"] = len(matched)
    if len(matched) < 20:
        row["error"] = "too_few_matches"
        return row
    src = np.float32([kp1[m.queryIdx].pt for m in matched])
    dst = np.float32([kp2[m.trainIdx].pt for m in matched])
    try:
        cv2.setRNGSeed(42)
        row["quality"] = _fit_and_check(src, dst, working_size, target_size, Options())
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main():
    cv2.setNumThreads(1)
    for stem in EXAMPLES:
        aoi = "_".join(stem.split("_")[:2])
        with rasterio.open(ROOT / "msi" / aoi / (stem + "_msi.tif")) as ds:
            source = ds.read()
            valid = ds.read_masks(4) > 0
        with rasterio.open(ROOT / "crops" / aoi / (stem + "_pan.tif")) as ds:
            pan = ds.read(1)
        target = np.asarray(Image.open(ROOT / "pansharpened_he" / aoi / (stem + "_rgb.png")).convert("RGB"))
        rgb = stretch_rgb(np.moveaxis(source[[2, 1, 0]], 0, -1))
        sh, sw = rgb.shape[:2]
        th, tw = target.shape[:2]
        small = cv2.resize(target, (sw, sh), interpolation=cv2.INTER_AREA)
        content = np.max(target, axis=2) > 2
        ys, xs = np.nonzero(content)
        row = {
            "id": stem,
            "msi": describe(source),
            "pan": describe(pan),
            "rgb": describe(target),
            "rgb_content_fraction": float(content.mean()),
            "rgb_content_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "trials": [],
        }
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        strategies = {
            "gray": (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)),
            "green": (rgb[:, :, 1], small[:, :, 1]),
            "green_clahe": (clahe.apply(rgb[:, :, 1]), clahe.apply(small[:, :, 1])),
            "gray_clahe": (clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)), clahe.apply(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))),
        }
        for name, (source_gray, target_gray) in strategies.items():
            for contrast, ratio in ((0.02, 0.75), (0.005, 0.8), (0.001, 0.85)):
                result = fit(source_gray, target_gray, (sw, sh), (tw, th), contrast, ratio)
                result["strategy"] = name
                row["trials"].append(result)
                if "quality" in result:
                    h = np.array(result["quality"]["H_native_to_target"])
                    _, mapped_valid = warp_nir(source[3], valid, h, (tw, th))
                    result["valid_fraction"] = float(mapped_valid.mean())
                    result["rgb_content_supported_fraction"] = float(mapped_valid[content].mean()) if content.any() else None
                    break
            if "quality" in row["trials"][-1] and stem in EXAMPLES[:2]:
                break
        panels = [show_panel(target), show_panel(rgb), show_panel(pan), show_panel(source[3])]
        height = max(p.height for p in panels)
        canvas = Image.new("RGB", (1600, height + 32), "white")
        drawer = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, ("Current RGB", "Native MSI visible", "PAN", "Native NIR"))):
            canvas.paste(panel, (400 * index, 32))
            drawer.text((400 * index + 5, 8), label, fill="black")
        canvas.save(OUTPUT / (stem + "_contact.png"))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

"""Read-only cross-band geometry checks for the user's selected example."""

from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"E:\Dataset\21century\registration_output\rectified\4\region_01\image_1_rectified.tif")


def display(band):
    lo, hi = np.percentile(band, [1, 99])
    return np.clip((band.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1).astype(np.float32)


def magnitude(band):
    band = cv2.GaussianBlur(display(band), (0, 0), 1)
    dx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def compare(reference, moving):
    ref = magnitude(reference)
    mov = magnitude(moving)
    hann = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref, mov, hann)
    warp = np.eye(2, 3, dtype=np.float32)
    record = {"phase_moving_relative_to_reference_xy": list(shift),
              "phase_response": float(response)}
    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                       100, 1e-6), None, 5)
        record.update(ecc_correlation=float(cc),
                      ecc_moving_relative_to_reference_xy=warp[:, 2].tolist())
    except cv2.error as exc:
        record["ecc_error"] = str(exc)
    return record


def sift_translation(reference, moving):
    # Match structural features; robust consensus rejects spectral differences.
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    a = (display(reference) * 255).astype(np.uint8)
    b = (display(moving) * 255).astype(np.uint8)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return {"matches": 0}
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    if len(good) < 4:
        return {"matches": len(good)}
    affine, valid = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                              ransacReprojThreshold=1.5)
    if affine is None:
        return {"matches": len(good)}
    keep = valid.ravel().astype(bool)
    residual = pb[keep] - pa[keep]
    return {"matches": len(good), "inliers": int(keep.sum()),
            "median_moving_relative_to_reference_xy": np.median(residual, axis=0).tolist(),
            "displacement_iqr_xy": (np.percentile(residual, 75, axis=0)
                                     - np.percentile(residual, 25, axis=0)).tolist(),
            "affine_reference_to_moving": affine.tolist()}


def main():
    # The user supplied the exact tile identity; no screenshot localization needed.
    with rasterio.open(SOURCE) as source:
        bands = source.read(window=Window(5120, 3072, 1024, 1024))
    report = {"source": str(SOURCE), "tile_bounds_xywh": [5120, 3072, 1024, 1024],
              "interpretation": "Positive dx means the NIR feature is to the right of the RGB feature. These cross-spectral estimators are diagnostics, not a registration ground truth.",
              "whole_tile": {}, "subregions": [], "coarse_scales": []}
    for index, label in [(2, "red_to_nir"), (1, "green_to_nir"), (0, "blue_to_nir")]:
        record = compare(bands[index], bands[3])
        record["sift"] = sift_translation(bands[index], bands[3])
        report["whole_tile"][label] = record
    for y in [0, 256, 512]:
        for x in [0, 256, 512]:
            roi = bands[:, y:y + 512, x:x + 512]
            report["subregions"].append({"tile_roi_xywh": [x, y, 512, 512],
                                         **compare(roi[2], roi[3])})
    for factor in [2, 4, 8, 16]:
        coarse = [cv2.resize(band.astype(np.float32), (1024 // factor,) * 2,
                             interpolation=cv2.INTER_AREA) for band in bands]
        record = compare(coarse[2], coarse[3])
        record["downsample_factor"] = factor
        record["phase_shift_in_source_pixels"] = (
            np.array(record["phase_moving_relative_to_reference_xy"]) * factor).tolist()
        if "ecc_moving_relative_to_reference_xy" in record:
            record["ecc_shift_in_source_pixels"] = (
                np.array(record["ecc_moving_relative_to_reference_xy"]) * factor).tolist()
        report["coarse_scales"].append(record)
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "locate_example.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

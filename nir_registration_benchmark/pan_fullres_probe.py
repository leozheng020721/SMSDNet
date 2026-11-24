"""Probe PAN-native detection resolution without modifying any dataset products.

The ORB500 cross-check setup follows the detection grid and matcher in
https://github.com/centreborelli/shadow-eo-dataset/blob/main/pansharpen.py .
Radiometric preprocessing is kept identical to the current comparison; this
does not claim to recreate the original metadata-dependent TOA corrections.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.errors import NotGeoreferencedWarning

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path.insert(0, str(REPOSITORY))

from align_seo_nir import resize_coordinate_matrix, stretch_rgb
from nir_registration_metrics import evaluate_registration, fit_and_validate_spatial


METHODS = ("orb500_crosscheck", "orb8000_crosscheck", "orb8000_mutual_ratio", "sift5000_mutual_ratio")


def selected_records(path, keys):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("key") in keys:
                    records[record["key"]] = record
    return records


def select_probe_samples(samples):
    selected = []
    for city, acquisition in (("JAX", 8), ("JAX", 9), ("JAX", 15),
                              ("OMA", 12), ("UCSD", 30), ("UCSD", 3)):
        matches = [sample for sample in samples if sample["city"] == city
                   and sample["cohort"] == "failed_matching"
                   and int(sample["key"].rsplit("_", 1)[1]) == acquisition]
        if not matches:
            raise ValueError(f"No formal failed sample for {city} acquisition {acquisition}")
        selected.append(matches[0])
    for city in ("JAX", "OMA", "UCSD"):
        selected.append(next(sample for sample in samples if sample["city"] == city
                             and sample["cohort"] == "accepted_control"))
    return selected


def read_pair(root, key):
    relative = Path(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(root / "msi" / relative.parent / (relative.name + "_msi.tif")) as dataset:
            visible = stretch_rgb(np.moveaxis(dataset.read([3, 2, 1]), 0, -1))
            valid = dataset.read_masks(4) > 0
        with rasterio.open(root / "crops" / relative.parent / (relative.name + "_pan.tif")) as dataset:
            pan = dataset.read(1)
    with Image.open(root / "pansharpened_he" / relative.parent / (relative.name + "_rgb.png")) as image:
        target = np.asarray(image.convert("RGB"))
    finite = np.isfinite(pan)
    low, high = np.percentile(pan[finite], (1, 99))
    pan = np.uint8(np.clip((np.where(finite, pan, low).astype(np.float32) - low)
                          / max(float(high - low), 1), 0, 1) * 255)
    return visible, valid, pan, target


def match_full_resolution(source_rgb, pan, method):
    """Return native-MSI and native-PAN coordinates from PAN-grid detections."""
    source_size = (source_rgb.shape[1], source_rgb.shape[0])
    pan_size = (pan.shape[1], pan.shape[0])
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    enlarged = cv2.resize(gray, pan_size, interpolation=cv2.INTER_CUBIC)
    if method.startswith("orb"):
        features = 500 if method.startswith("orb500_") else 8000
        detector = cv2.ORB_create(nfeatures=features)
        norm = cv2.NORM_HAMMING
    elif method == "sift5000_mutual_ratio":
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
        features, norm = 5000, cv2.NORM_L2
    else:
        raise ValueError(f"Unknown probe method: {method}")
    cv2.setRNGSeed(42)
    kp_source, des_source = detector.detectAndCompute(enlarged, None)
    kp_target, des_target = detector.detectAndCompute(pan, None)
    stats = {"detection_size_wh": list(pan_size), "feature_limit": features,
             "source_keypoints": len(kp_source), "pan_keypoints": len(kp_target),
             "matcher": "crossCheck" if method.endswith("crosscheck") else "mutual ratio"}
    empty = np.empty((0, 2), np.float64)
    if des_source is None or des_target is None or min(len(des_source), len(des_target)) < 2:
        return empty, empty.copy(), {**stats, "match_count": 0}
    if method.endswith("crosscheck"):
        matches = cv2.BFMatcher(norm, crossCheck=True).match(des_source, des_target)
    else:
        ratio = 0.8 if method.startswith("orb") else 0.75
        stats["ratio"] = ratio
        matcher = cv2.BFMatcher(norm)
        forward = [pair[0] for pair in matcher.knnMatch(des_source, des_target, k=2)
                   if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
        reverse = {(pair[0].trainIdx, pair[0].queryIdx)
                   for pair in matcher.knnMatch(des_target, des_source, k=2)
                   if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
        matches = [match for match in forward if (match.queryIdx, match.trainIdx) in reverse]
    source_points = np.float64([kp_source[match.queryIdx].pt for match in matches]).reshape(-1, 2)
    pan_points = np.float64([kp_target[match.trainIdx].pt for match in matches]).reshape(-1, 2)
    inverse = np.linalg.inv(resize_coordinate_matrix(source_size, pan_size))
    if len(source_points):
        source_points = cv2.perspectiveTransform(source_points[:, None], inverse)[:, 0]
    stats["match_count"] = len(matches)
    return source_points, pan_points, stats


def verified_bridge(formal, cached, source_size, pan_size, target_size):
    grid = formal.get("grid", {})
    if grid.get("verified") and grid.get("spatial_validation", {}).get("accepted"):
        return {"accepted": True, "source": "formal_v2 spatial verification",
                "H_pan_to_rgb": grid["H_pan_to_rgb"], "qa": grid["spatial_validation"]["metrics"]}
    cached_grid = cached.get("grid", {})
    src = np.asarray(cached_grid.get("points_src_pan", []), np.float64).reshape(-1, 2)
    dst = np.asarray(cached_grid.get("points_dst_rgb", []), np.float64).reshape(-1, 2)
    pan_to_work = resize_coordinate_matrix(pan_size, source_size)
    work_points = cv2.perspectiveTransform(src[:, None], pan_to_work)[:, 0] if len(src) else src
    fitted = fit_and_validate_spatial(work_points, dst, source_size, target_size)
    result = {"accepted": fitted["accepted"], "source": "independent spatial revalidation of cached grid points",
              "reason": fitted["reason"], "qa": fitted["metrics"]}
    if fitted["accepted"]:
        result["H_pan_to_rgb"] = (np.asarray(fitted["H_native_to_target"]) @ pan_to_work).tolist()
    return result


def preview(path, source, pan, target, methods):
    passed = [result for result in methods.values() if result.get("accepted")]
    best = min(passed, key=lambda result: result["metrics"]["holdout_median_msi_px"]) if passed else None
    size = (target.shape[1], target.shape[0])
    if best:
        aligned = cv2.warpPerspective(source, np.asarray(best["H_native_to_target"]), size)
        edges1 = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 50, 120)
        edges2 = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), 50, 120)
        diagnostic = np.stack((edges2, edges1, edges1), axis=-1)
        diagnostic_label = "RGB cyan / aligned MSI red edges"
    else:
        diagnostic = cv2.resize(source, size, interpolation=cv2.INTER_CUBIC)
        diagnostic_label = "MSI resize only (NOT registered)"
    width, height = 320, max(1, round(target.shape[0] * 320 / target.shape[1]))
    canvas = Image.new("RGB", (width * 3, height + 30), "white")
    drawer = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate((("Current RGB", target), ("Native PAN", pan),
                                            (diagnostic_label, diagnostic))):
        canvas.paste(Image.fromarray(image).convert("RGB").resize((width, height)), (index * width, 30))
        drawer.text((index * width + 3, 7), label, fill="black")
    canvas.save(path, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--samples", type=Path, default=HERE / "comparison_v1" / "samples.json")
    parser.add_argument("--formal-results", type=Path, default=HERE / "formal_v2" / "results.jsonl")
    parser.add_argument("--pan-cache", type=Path, default=HERE / "comparison_v1" / "pan_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "pan_fullres_probe")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    samples = select_probe_samples(json.loads(args.samples.read_text(encoding="utf-8")))
    keys = {sample["key"] for sample in samples}
    formal = selected_records(args.formal_results, keys)
    cached = selected_records(args.pan_cache, keys)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise RuntimeError("Existing probe results found; choose a fresh output directory")
    config = {"methods": list(METHODS), "source_root": str(args.root), "samples": samples,
              "detection_grid": "original PAN resolution; MSI gray enlarged with cubic interpolation",
              "radiometry": "same 1/99 percentile visible/PAN preprocessing as formal comparison",
              "qa": "unchanged fit_and_validate_spatial defaults; target points bridged into original RGB grid",
              "threads": 1, "gdal_cache_bytes": 32 * 1024 * 1024,
              "source_sha256": {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), REPOSITORY / "nir_registration_metrics.py",
                                             REPOSITORY / "align_seo_nir.py")},
              "official_orb_reference": "https://github.com/centreborelli/shadow-eo-dataset/blob/main/pansharpen.py"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    previewed_failed_cities = set()
    with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
        for index, sample in enumerate(samples, 1):
            started = time.monotonic()
            source, valid, pan, target = read_pair(args.root, sample["key"])
            source_size = (source.shape[1], source.shape[0])
            pan_size, target_size = (pan.shape[1], pan.shape[0]), (target.shape[1], target.shape[0])
            bridge = verified_bridge(formal.get(sample["key"], {}), cached.get(sample["key"], {}),
                                     source_size, pan_size, target_size)
            record = {**sample, "bridge": bridge, "methods": {},
                      "formal_v2_methods": {name: {"accepted": result.get("accepted"), "reason": result.get("reason")}
                                            for name, result in formal.get(sample["key"], {}).get("methods", {}).items()}}
            for method in METHODS:
                method_started = time.monotonic()
                try:
                    src, dst_pan, stats = match_full_resolution(source, pan, method)
                    if bridge["accepted"]:
                        h_grid = np.asarray(bridge["H_pan_to_rgb"], np.float64)
                        dst_rgb = cv2.perspectiveTransform(dst_pan[:, None], h_grid)[:, 0] if len(dst_pan) else dst_pan
                        fitted = fit_and_validate_spatial(src, dst_rgb, source_size, target_size)
                        if fitted["accepted"]:
                            fitted["image_metrics"] = evaluate_registration(
                                source, target, np.asarray(fitted["H_native_to_target"]), source_valid=valid)
                            if fitted["image_metrics"].get("valid_fraction", 0) < 0.1:
                                fitted.update(accepted=False, reason="Less than 10 percent valid NIR coverage")
                    else:
                        fitted = {"accepted": False, "reason": "PAN/RGB bridge failed spatial QA"}
                    record["methods"][method] = {**fitted, **stats}
                except Exception as exc:
                    record["methods"][method] = {"accepted": False, "execution_error": f"{type(exc).__name__}: {exc}"}
                record["methods"][method]["seconds"] = round(time.monotonic() - method_started, 3)
                gc.collect()
            if sample["cohort"] == "accepted_control" or sample["city"] not in previewed_failed_cities:
                try:
                    preview(args.output / (sample["key"].replace("/", "__") + ".jpg"), source, pan, target, record["methods"])
                except Exception as exc:
                    record["preview_error"] = f"{type(exc).__name__}: {exc}"
                if sample["cohort"] == "failed_matching":
                    previewed_failed_cities.add(sample["city"])
            record["seconds"] = round(time.monotonic() - started, 3)
            records.append(record)
            with (args.output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({"done": index, "key": sample["key"], "bridge": bridge["accepted"],
                              "accepted": {name: result.get("accepted") for name, result in record["methods"].items()},
                              "match_counts": {name: result.get("match_count") for name, result in record["methods"].items()},
                              "seconds": record["seconds"]}), flush=True)
            del source, valid, pan, target
            gc.collect()
    summary = {"samples": len(records), "by_cohort": {},
               "execution_errors": sum("execution_error" in result for record in records for result in record["methods"].values()),
               "caution": "Small diagnostic sample; PAN ensemble selection is not independent external validation."}
    for cohort in ("failed_matching", "accepted_control"):
        subset = [record for record in records if record["cohort"] == cohort]
        summary["by_cohort"][cohort] = {"n": len(subset), **{
            method: sum(bool(record["methods"][method].get("accepted")) for record in subset) for method in METHODS}}
        summary["by_cohort"][cohort]["any_fullres_method"] = sum(
            any(result.get("accepted") for result in record["methods"].values()) for record in subset)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

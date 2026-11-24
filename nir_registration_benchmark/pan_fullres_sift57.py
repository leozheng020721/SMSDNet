"""Compare PAN-native SIFT on the frozen 57-case cohort, stage by stage."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path.insert(0, str(REPOSITORY))

from align_seo_nir import resize_coordinate_matrix
from nir_registration_metrics import (
    _balanced_subset,
    evaluate_registration,
    fit_and_validate_spatial,
    homography_geometry,
    registration_support_mask,
    spatial_holdout_split,
)
from nir_registration_benchmark.pan_fullres_probe import match_full_resolution, read_pair, selected_records


def bridge_record(formal):
    grid = formal.get("grid", {})
    spatial = grid.get("spatial_validation", {})
    fallback = grid.get("loftr_fallback", {})
    verified = bool(grid.get("verified") and (spatial.get("accepted") or fallback.get("accepted")))
    result = {"verified": verified, "formal_verified": bool(grid.get("verified")),
              "same_grid": grid.get("same_grid"), "spatial_validation": spatial,
              "loftr_fallback": fallback, "verification_method": grid.get("verification_method", "classical spatial QA"),
              "reason": None if verified else fallback.get("reason", spatial.get("reason", grid.get("reason")))}
    if verified:
        result["H_pan_to_rgb"] = grid["H_pan_to_rgb"]
    return result


def composed_holdout(src, dst_rgb, h, source_size, target_size):
    """Evaluate the explicit composition on the exact shared spatial protocol."""
    sw, sh = source_size
    tw, th = target_size
    inside = ((src[:, 0] >= 0) & (src[:, 0] < sw) & (src[:, 1] >= 0) & (src[:, 1] < sh)
              & (dst_rgb[:, 0] >= 0) & (dst_rgb[:, 0] < tw)
              & (dst_rgb[:, 1] >= 0) & (dst_rgb[:, 1] < th))
    src, dst_rgb = src[inside], dst_rgb[inside]
    indices = _balanced_subset(src, source_size, (4, 4), 3000, None, 42)
    src, dst_rgb = src[indices], dst_rgb[indices]
    split = spatial_holdout_split(src, dst_rgb, source_size)
    holdout = split["test_indices"]
    result = {"accepted": False, "holdout_count": len(holdout), "units": "native MSI pixels"}
    if len(holdout) < 8:
        result["reason"] = "Too few untouched spatial holdout points"
        return result
    target_to_work = resize_coordinate_matrix(target_size, source_size)
    prediction = cv2.perspectiveTransform(src[holdout, None], target_to_work @ h)[:, 0]
    truth = cv2.perspectiveTransform(dst_rgb[holdout, None], target_to_work)[:, 0]
    error = np.linalg.norm(prediction - truth, axis=1)
    result.update(holdout_median_msi_px=float(np.median(error)),
                  holdout_p90_msi_px=float(np.percentile(error, 90)),
                  holdout_inlier_fraction=float(np.mean(error <= 1.5)))
    result["accepted"] = bool(result["holdout_median_msi_px"] <= 0.75
                              and result["holdout_p90_msi_px"] <= 3
                              and result["holdout_inlier_fraction"] >= 0.65)
    result["reason"] = None if result["accepted"] else "Explicit composition fails spatial holdout error thresholds"
    return result


def assess_case(sample, root, formal):
    start = time.monotonic()
    source, valid, pan, target = read_pair(root, sample["key"])
    source_size = (source.shape[1], source.shape[0])
    pan_size = (pan.shape[1], pan.shape[0])
    target_size = (target.shape[1], target.shape[0])
    src, dst_pan, extraction = match_full_resolution(source, pan, "sift5000_mutual_ratio")
    pan_fit = fit_and_validate_spatial(src, dst_pan, source_size, pan_size)
    if pan_fit["accepted"]:
        pan_h = np.asarray(pan_fit["H_native_to_target"])
        pan_fit["valid_fraction"] = float(registration_support_mask(valid, pan_h, pan_size).mean())
        pan_fit["all_source_valid_fraction"] = float(registration_support_mask(np.ones_like(valid), pan_h, pan_size).mean())
        if pan_fit["valid_fraction"] < 0.1:
            pan_fit.update(accepted=False, reason="Less than 10 percent valid NIR coverage on native PAN")
    bridge = bridge_record(formal)
    rgb_fit = {"accepted": False, "assessed": False, "reason": "Formal PAN/RGB bridge is not spatially verified"}
    composed = {"accepted": False, "assessed": False, "H_native_to_target": None,
                "reason": "MSI/PAN strict QA or formal PAN/RGB bridge did not pass"}
    if bridge["verified"]:
        grid_h = np.asarray(bridge["H_pan_to_rgb"], np.float64)
        dst_rgb = cv2.perspectiveTransform(dst_pan[:, None], grid_h)[:, 0] if len(dst_pan) else dst_pan
        rgb_fit = fit_and_validate_spatial(src, dst_rgb, source_size, target_size)
        rgb_fit["assessed"] = True
        if rgb_fit["accepted"]:
            rgb_fit["valid_fraction"] = float(registration_support_mask(
                valid, np.asarray(rgb_fit["H_native_to_target"]), target_size).mean())
            if rgb_fit["valid_fraction"] < 0.1:
                rgb_fit.update(accepted=False, reason="Less than 10 percent valid NIR coverage on RGB")
        if pan_fit["accepted"]:
            h = grid_h @ np.asarray(pan_fit["H_native_to_target"])
            geometry = homography_geometry(h, source_size, target_size)
            composed.update(assessed=True, geometry=geometry, candidate_H_native_to_rgb=h.tolist())
            if geometry["valid"]:
                composed["spatial_holdout"] = composed_holdout(src, dst_rgb, h, source_size, target_size)
                composed["valid_fraction"] = float(registration_support_mask(valid, h, target_size).mean())
                composed["accepted"] = bool(rgb_fit["accepted"] and composed["spatial_holdout"]["accepted"]
                                              and composed["valid_fraction"] >= 0.1)
                if composed["accepted"]:
                    composed.update(H_native_to_target=h.tolist(), reason=None)
                    composed["image_metrics"] = evaluate_registration(source, target, h, source_valid=valid)
                else:
                    composed["reason"] = (rgb_fit.get("reason") if not rgb_fit["accepted"]
                                           else composed["spatial_holdout"].get("reason")
                                           or "Less than 10 percent valid NIR coverage on RGB")
            else:
                composed["reason"] = geometry["reason"]
    record = {**sample, "extraction": extraction, "source_size_wh": source_size, "pan_size_wh": pan_size,
              "target_size_wh": target_size, "target_nonblack_fraction": float(np.any(target > 0, axis=2).mean()),
              "pan_only": pan_fit, "formal_bridge": bridge, "rgb_correspondence_fit": rgb_fit,
              "composed_rgb": composed, "seconds": round(time.monotonic() - start, 3),
              "caution": "Spatial matcher holdout is internal evidence, not independent external ground truth."}
    return record, (source, pan, target)


def save_preview(path, record, arrays):
    source, pan, target = arrays
    pan_fit, composed = record["pan_only"], record["composed_rgb"]
    pan_size = (pan.shape[1], pan.shape[0])
    if pan_fit.get("accepted"):
        pan_aligned = cv2.warpPerspective(source, np.asarray(pan_fit["H_native_to_target"]), pan_size)
        pan_label = "MSI -> PAN PASS"
    else:
        pan_aligned = cv2.resize(source, pan_size, interpolation=cv2.INTER_CUBIC)
        pan_label = "MSI resize only: PAN QA rejected"
    if composed.get("accepted"):
        rgb_aligned = cv2.warpPerspective(source, np.asarray(composed["H_native_to_target"]),
                                           (target.shape[1], target.shape[0]))
        rgb_label = "MSI -> current RGB PASS"
    else:
        rgb_aligned = np.zeros_like(target)
        rgb_label = "Current RGB mapping not accepted"
    width, height = 320, max(1, round(target.shape[0] * 320 / target.shape[1]))
    canvas = Image.new("RGB", (width * 4, height + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate((("Current RGB", target), ("Native PAN", pan),
                                            (pan_label, pan_aligned), (rgb_label, rgb_aligned))):
        canvas.paste(Image.fromarray(image).convert("RGB").resize((width, height)), (index * width, 30))
        draw.text((index * width + 3, 7), label, fill="black")
    canvas.save(path, quality=92)


def compact(record):
    return {"key": record["key"], "city": record["city"], "cohort": record["cohort"],
            "pan_only": bool(record.get("pan_only", {}).get("accepted")),
            "formal_bridge": bool(record.get("formal_bridge", {}).get("verified")),
            "rgb_fit": bool(record.get("rgb_correspondence_fit", {}).get("accepted")),
            "end_to_end": bool(record.get("composed_rgb", {}).get("accepted")),
            "execution_error": record.get("execution_error")}


def summarize(records):
    summary = {"sample_count": len(records), "execution_errors": sum(bool(r["execution_error"]) for r in records),
               "by_cohort": {}, "by_city": {}}
    for field in ("cohort", "city"):
        for group in sorted({record[field] for record in records}):
            selected = [record for record in records if record[field] == group]
            summary["by_" + field][group] = {"n": len(selected), **{
                stage: sum(record[stage] for record in selected)
                for stage in ("pan_only", "formal_bridge", "rgb_fit", "end_to_end")},
                "pan_pass_bridge_unverified": [record["key"] for record in selected
                                                if record["pan_only"] and not record["formal_bridge"]]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--samples", type=Path, default=HERE / "comparison_v1" / "samples.json")
    parser.add_argument("--formal-results", type=Path, default=HERE / "formal_v2" / "results.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "pan_fullres_sift57")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    if len(samples) != 57 or Counter(sample["cohort"] for sample in samples) != {"failed_matching": 48, "accepted_control": 9}:
        raise ValueError("Expected the same frozen 48-failure/9-control cohort")
    formal = selected_records(args.formal_results, {sample["key"] for sample in samples})
    if len(formal) != 57:
        raise ValueError("The complete formal_v2 result snapshot is required")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "results.jsonl"
    if manifest.exists():
        raise ValueError("Existing results found; choose a fresh output directory")
    paths = (Path(__file__), HERE / "pan_fullres_probe.py", REPOSITORY / "nir_registration_metrics.py",
             REPOSITORY / "align_seo_nir.py", args.samples, args.formal_results)
    config = {"method": "native PAN SIFT5000, mutual ratio 0.75", "root": str(args.root),
              "samples": samples, "gdal_cache_bytes": 32 * 1024 * 1024, "threads": 1,
              "strict_qa": "Frozen spatial defaults plus >=0.1 valid support for each accepted grid",
              "end_to_end": "PAN QA + same formal bridge + RGB refit QA + explicit composition heldout QA",
              "sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    previews = args.output / "previews"
    previews.mkdir(exist_ok=True)
    compact_records = []
    failed_preview_cities = set()
    started = time.monotonic()
    with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
        for index, sample in enumerate(samples, 1):
            arrays = None
            try:
                record, arrays = assess_case(sample, args.root, formal[sample["key"]])
            except Exception as exc:
                record = {**sample, "execution_error": f"{type(exc).__name__}: {exc}"}
            if arrays is not None and (record["cohort"] == "accepted_control"
                                       or record["pan_only"]["accepted"]
                                       or record["city"] not in failed_preview_cities):
                try:
                    save_preview(previews / (sample["key"].replace("/", "__") + ".jpg"), record, arrays)
                    if record["cohort"] == "failed_matching":
                        failed_preview_cities.add(record["city"])
                except Exception as exc:
                    record["preview_error"] = f"{type(exc).__name__}: {exc}"
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            compact_records.append(compact(record))
            print(json.dumps({"done": index, **compact_records[-1], "seconds": record.get("seconds")}), flush=True)
            if index % 10 == 0 or index == len(samples):
                snapshot = {"done": index, "elapsed_seconds": round(time.monotonic() - started, 2),
                            **summarize(compact_records)}
                (args.output / "summary.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                print(json.dumps({"event": "snapshot", **snapshot}), flush=True)
            del record, arrays
            gc.collect()


if __name__ == "__main__":
    main()

"""Compare direct LoFTR and PAN-assisted registration without changing data."""

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from align_seo_nir import load_records, resize_coordinate_matrix, stretch_rgb


HERE = Path(__file__).resolve().parent
VENDOR = HERE / "nir_registration_benchmark" / "vendor"
sys.path.insert(0, str(VENDOR))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def select_samples(root, per_city=16, controls_per_city=3, seed=20261109):
    records = load_records(root / "nir_aligned_v1" / "registration.jsonl")
    rng = random.Random(seed)
    selected = []
    for city in ("JAX", "OMA", "UCSD"):
        groups = defaultdict(list)
        controls = []
        for key, record in sorted(records.items()):
            if not key.startswith(city + "_"):
                continue
            last_error = record.get("error", "").split(";")[-1]
            if record["status"] == "failed" and (
                    "mutual matches" in last_error or "Insufficient visible texture" in last_error):
                groups[key.rsplit("_", 1)[1]].append(key)
            elif record["status"] == "ok" and record.get("valid_fraction", 0) >= 0.95:
                controls.append(key)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < per_city and any(groups.values()):
            for acquisition in sorted(groups):
                if groups[acquisition] and len(chosen) < per_city:
                    chosen.append(groups[acquisition].pop())
        for key in chosen:
            selected.append({"key": key, "cohort": "failed_matching", "city": city})
        rng.shuffle(controls)
        for key in controls[:controls_per_city]:
            selected.append({"key": key, "cohort": "accepted_control", "city": city,
                             "reference_H": records[key]["H_native_to_target"]})
    return selected


def load_pair(root, key):
    relative = Path(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
            with rasterio.open(root / "msi" / relative.parent / (relative.name + "_msi.tif")) as ds:
                msi = ds.read()
                valid = ds.read_masks(4) > 0
            with rasterio.open(root / "crops" / relative.parent / (relative.name + "_pan.tif")) as ds:
                pan = ds.read(1)
    rgb = np.asarray(Image.open(root / "pansharpened_he" / relative.parent /
                               (relative.name + "_rgb.png")).convert("RGB"))
    visible = stretch_rgb(np.moveaxis(msi[[2, 1, 0]], 0, -1))
    return visible, pan, rgb, valid


def pan_display(pan):
    finite = np.isfinite(pan)
    low, high = np.percentile(pan[finite], (1, 99))
    return np.uint8(np.clip((pan.astype(np.float32) - low) / max(high - low, 1), 0, 1) * 255)


class LoFTRMatcher:
    def __init__(self, weights, edge=640, confidence=0.2):
        import torch
        from kornia.feature import LoFTR
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.edge = edge
        self.confidence = confidence
        torch.set_num_threads(4)
        self.model = LoFTR(pretrained=None).eval()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self.model = self.model.to(self.device)

    def prepare(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        height, width = gray.shape
        scale = self.edge / max(height, width)
        size = (max(32, round(width * scale / 8) * 8),
                max(32, round(height * scale / 8) * 8))
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        tensor = self.torch.from_numpy(resized.astype(np.float32) / 255)[None, None].to(self.device)
        return tensor, np.linalg.inv(resize_coordinate_matrix((width, height), size))

    def match(self, source, target):
        first, inv_first = self.prepare(source)
        second, inv_second = self.prepare(target)
        with self.torch.inference_mode():
            result = self.model({"image0": first, "image1": second})
        confidence = result["confidence"].cpu().numpy()
        keep = confidence >= self.confidence
        src = result["keypoints0"].cpu().numpy()[keep]
        dst = result["keypoints1"].cpu().numpy()[keep]
        if len(src):
            src = cv2.perspectiveTransform(src[:, None, :], inv_first)[:, 0]
            dst = cv2.perspectiveTransform(dst[:, None, :], inv_second)[:, 0]
        return src, dst, confidence[keep]


def slim(record):
    return {key: value for key, value in record.items()
            if not key.startswith("points_") and key not in ("attempts", "qa", "H_native_to_pan_raw")}


def assess(src, dst, source, target, valid, confidence=None):
    from nir_registration_metrics import evaluate_registration, fit_and_validate_spatial
    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    fitted = fit_and_validate_spatial(
        src, dst, (source.shape[1], source.shape[0]), (target.shape[1], target.shape[0]),
        confidence=confidence)
    if fitted.get("H_native_to_target") is not None:
        h = np.asarray(fitted["H_native_to_target"], np.float64)
        fitted["image_metrics"] = evaluate_registration(source, target, h, source_valid=valid)
        if fitted["image_metrics"].get("valid_fraction", 0) < 0.1:
            fitted.update(accepted=False, reason="Less than 10 percent valid NIR coverage")
    return fitted


def result_h(result):
    value = result.get("H_native_to_target")
    return None if value is None else np.asarray(value, np.float64)


def candidate_score(result):
    metrics = result.get("metrics", {})
    median = metrics.get("holdout_median_msi_px", metrics.get("holdout_median_source_px", 1e9))
    return (not result.get("accepted", False), median)


def assess_loftr(src, dst, confidence, source, target, valid):
    attempts = []
    for threshold in (0.5, 0.2):
        keep = confidence >= threshold
        result = assess(src[keep], dst[keep], source, target, valid, confidence[keep])
        result.update(confidence_threshold=threshold, match_count=int(keep.sum()))
        attempts.append(result)
    selected = min(attempts, key=candidate_score)
    selected["threshold_attempts"] = [{"threshold": r["confidence_threshold"], "accepted": r["accepted"],
                                       "reason": r.get("reason"), "matches": r["match_count"]} for r in attempts]
    return selected


def check_grid_points(pan_points, rgb_points, confidence, source, pan, target):
    from nir_registration_metrics import fit_and_validate_spatial
    pan_points = np.asarray(pan_points, np.float32).reshape(-1, 2)
    rgb_points = np.asarray(rgb_points, np.float32).reshape(-1, 2)
    pan_to_work = resize_coordinate_matrix((pan.shape[1], pan.shape[0]), (source.shape[1], source.shape[0]))
    working = cv2.perspectiveTransform(pan_points[:, None], pan_to_work)[:, 0] if len(pan_points) else pan_points
    fitted = fit_and_validate_spatial(working, rgb_points, (source.shape[1], source.shape[0]),
                                      (target.shape[1], target.shape[0]), confidence=confidence)
    if fitted.get("accepted"):
        fitted["H_pan_to_rgb"] = np.asarray(fitted["H_native_to_target"]) @ pan_to_work
    return fitted


def runtime_failure(exc):
    return {"accepted": False, "reason": f"{type(exc).__name__}: {exc}", "runtime_error": True}


def read_cached_pair(path, key):
    needle = json.dumps(key)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if needle not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("key") == key:
                return record
    return None


def make_preview(path, source, pan, target, methods):
    target_size = (target.shape[1], target.shape[0])
    panels = [("Current RGB", target), ("PAN", pan_display(pan))]
    for name, result in methods.items():
        h = result_h(result)
        if h is None:
            aligned = np.zeros_like(target)
        else:
            aligned = cv2.warpPerspective(source, h, target_size)
        panels.append((name + (" PASS" if result.get("accepted") else " REJECT"), aligned))
    width = 320
    height = max(1, round(target.shape[0] * width / target.shape[1]))
    canvas = Image.new("RGB", (width * len(panels), height + 30), "white")
    drawer = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        panel = Image.fromarray(panel).convert("RGB").resize((width, height))
        canvas.paste(panel, (index * width, 30))
        drawer.text((index * width + 4, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def compare_one(sample, root, matcher, pan_cache=None):
    from nir_registration_pan import estimate_pan_candidates
    started = time.monotonic()
    source, pan, target, valid = load_pair(root, sample["key"])
    methods = {}
    try:
        src, dst, confidence = matcher.match(source, target)
        methods["direct_loftr"] = assess_loftr(src, dst, confidence, source, target, valid)
    except Exception as exc:
        methods["direct_loftr"] = runtime_failure(exc)
    try:
        pan_candidates = read_cached_pair(pan_cache, sample["key"]) if pan_cache else None
        if pan_candidates is None:
            pan_candidates = estimate_pan_candidates(source, pan, target)
    except Exception as exc:
        pan_candidates = {"grid": {"verified": False, "runtime_failure": runtime_failure(exc)}, "candidates": []}
    grid = dict(pan_candidates["grid"])
    if grid.get("verified"):
        fitted = check_grid_points(grid["points_src_pan"], grid["points_dst_rgb"], None,
                                  source, pan, target)
        grid["spatial_validation"] = fitted
        grid["verified"] = fitted["accepted"]
        if fitted["accepted"]:
            grid["H_pan_to_rgb"] = fitted["H_pan_to_rgb"]
    if not grid.get("verified"):
        try:
            pan_points, rgb_points, grid_confidence = matcher.match(pan_display(pan), target)
            grid_fit = check_grid_points(pan_points, rgb_points, grid_confidence, source, pan, target)
        except Exception as exc:
            grid_fit = runtime_failure(exc)
        grid["loftr_fallback"] = grid_fit
        if grid_fit.get("accepted"):
            grid_h = grid_fit["H_pan_to_rgb"]
            displacement = np.linalg.norm(rgb_points - pan_points, axis=1)
            grid.update(verified=True, H_pan_to_rgb=grid_h,
                        same_grid=bool(np.median(displacement) <= 2 and np.percentile(displacement, 90) <= 5),
                        verification_method="loftr_spatial_holdout", points_src_pan=pan_points,
                        points_dst_rgb=rgb_points)
    classical = []
    for candidate in pan_candidates["candidates"]:
        if not grid.get("verified"):
            continue
        source_points = np.asarray(candidate["points_src_native"], np.float32).reshape(-1, 2)
        pan_points = np.asarray(candidate["points_dst_pan"], np.float32).reshape(-1, 2)
        rgb_points = cv2.perspectiveTransform(pan_points[:, None], np.asarray(grid["H_pan_to_rgb"], np.float64))[:, 0] if len(pan_points) else pan_points
        evaluation = assess(source_points, rgb_points,
                            source, target, valid)
        evaluation["strategy"] = candidate["strategy"]
        evaluation["match_count"] = len(candidate["points_src_native"])
        classical.append(evaluation)
    methods["pan_classical"] = min(classical, key=candidate_score) if classical else {
        "accepted": False, "reason": "PAN/RGB grid not verified or insufficient matches"}
    try:
        src_pan, dst_pan, confidence_pan = matcher.match(source, pan_display(pan))
        if grid.get("verified") and len(src_pan):
            grid_h = np.asarray(grid["H_pan_to_rgb"], np.float64)
            dst_rgb = cv2.perspectiveTransform(dst_pan[:, None, :], grid_h)[:, 0]
            methods["pan_loftr"] = assess_loftr(src_pan, dst_rgb, confidence_pan, source, target, valid)
        else:
            methods["pan_loftr"] = {"accepted": False, "reason": "PAN/RGB grid not verified", "match_count": len(src_pan)}
    except Exception as exc:
        methods["pan_loftr"] = runtime_failure(exc)
    methods["pan_any"] = min((methods["pan_classical"], methods["pan_loftr"]), key=candidate_score)
    if methods["direct_loftr"].get("accepted") and methods["pan_any"].get("accepted"):
        from nir_registration_metrics import evaluate_registration, registration_support_mask
        hs = [result_h(methods[name]) for name in ("direct_loftr", "pan_any")]
        common = (registration_support_mask(valid, hs[0], (target.shape[1], target.shape[0])) &
                  registration_support_mask(valid, hs[1], (target.shape[1], target.shape[0])))
        for name, h in zip(("direct_loftr", "pan_any"), hs):
            methods[name]["common_region_metrics"] = evaluate_registration(source, target, h,
                source_valid=valid, evaluation_mask=common)
    if sample.get("reference_H"):
        reference = np.asarray(sample["reference_H"], np.float64)
        yy, xx = np.meshgrid(np.linspace(0, source.shape[0] - 1, 8),
                             np.linspace(0, source.shape[1] - 1, 8), indexing="ij")
        probes = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)[:, None]
        expected = cv2.perspectiveTransform(probes, reference)[:, 0]
        for result in methods.values():
            h = result_h(result)
            if h is not None:
                predicted = cv2.perspectiveTransform(probes, h)[:, 0]
                result["reference_grid_median_rgb_px"] = float(np.median(np.linalg.norm(predicted - expected, axis=1)))
    return {**sample, "grid": slim(grid), "pan_candidate_count": len(pan_candidates["candidates"]),
            "methods": methods, "seconds": round(time.monotonic() - started, 3)}, (source, pan, target)


def summarize(records):
    summary = {"sample_count": len(records), "by_cohort": {}, "by_city": {}, "pan_grid_verified": 0}
    for record in records:
        summary["pan_grid_verified"] += bool(record.get("grid", {}).get("verified"))
    for field in ("cohort", "city"):
        grouped = defaultdict(list)
        for record in records:
            grouped[record[field]].append(record)
        for group, items in grouped.items():
            summary["by_" + field][group] = {"n": len(items)}
            for method in ("direct_loftr", "pan_classical", "pan_loftr", "pan_any"):
                passed = [r["methods"][method] for r in items if r.get("methods", {}).get(method, {}).get("accepted")]
                summary["by_" + field][group][method] = {"accepted": len(passed)}
                summary["by_" + field][group][method]["runtime_errors"] = sum(
                    bool(r.get("methods", {}).get(method, {}).get("runtime_error")) for r in items)
    summary["sample_runtime_errors"] = sum(bool(r.get("error")) for r in records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\Dataset\SEO"))
    parser.add_argument("--output", type=Path, default=HERE / "nir_registration_benchmark" / "comparison_v1")
    parser.add_argument("--weights", type=Path, default=HERE / "nir_registration_benchmark" / "loftr_outdoor.ckpt")
    parser.add_argument("--per-city", type=int, default=16)
    parser.add_argument("--controls-per-city", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pan-only", action="store_true")
    parser.add_argument("--pan-cache", type=Path)
    parser.add_argument("--samples-file", type=Path)
    parser.add_argument("--sample-key")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    samples = (json.loads(args.samples_file.read_text()) if args.samples_file else
               select_samples(args.root, args.per_city, args.controls_per_city))
    if args.sample_key:
        samples = [s for s in samples if s["key"] == args.sample_key]
        if len(samples) != 1:
            raise ValueError("Requested sample key must occur exactly once")
    (args.output / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    if args.limit:
        samples = samples[:args.limit]
    if args.prepare_only:
        print(json.dumps({"samples": len(samples), "cohorts": dict(Counter(s["cohort"] for s in samples))}))
        return
    if args.pan_only:
        from nir_registration_pan import estimate_pan_candidates
        cache_path = args.output / "pan_candidates.jsonl"
        existing = set()
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        existing.add(json.loads(line)["key"])
        for index, sample in enumerate(samples, 1):
            if sample["key"] in existing:
                continue
            started = time.monotonic()
            source, pan, target, valid = load_pair(args.root, sample["key"])
            record = {"key": sample["key"], **estimate_pan_candidates(source, pan, target)}
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=json_value) + "\n")
            print(json.dumps({"pan_done": index, "total": len(samples), "key": sample["key"],
                              "grid_verified": record["grid"].get("verified"),
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
        return
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    config = {"weights_sha256": weights_hash, "loftr_edge": 640, "confidence_candidates": [0.5, 0.2],
              "evaluation": "spatial block holdout, no refit using validation points",
              "source_root": str(args.root), "sample_count": len(samples),
              "sample_keys": [s["key"] for s in samples],
              "validation_thresholds": {"median_msi_px": 0.75, "p90_msi_px": 3.0,
                                         "holdout_fraction": 0.65, "min_hull_coverage": 0.05,
                                         "min_valid_fraction": 0.1, "source_guard_px": 4},
              "code_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in
                              ("compare_seo_nir_registration.py", "nir_registration_metrics.py", "nir_registration_pan.py")}}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Benchmark configuration differs; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    matcher = LoFTRMatcher(args.weights)
    pan_cache = args.pan_cache
    manifest = args.output / "results.jsonl"
    prior = load_records(manifest)
    results = []
    for index, sample in enumerate(samples, 1):
        if sample["key"] in prior and not prior[sample["key"]].get("error"):
            results.append(prior[sample["key"]])
            continue
        try:
            record, arrays = compare_one(sample, args.root, matcher, pan_cache)
            try:
                make_preview(args.output / "previews" / (sample["key"].replace("/", "__") + ".jpg"),
                             *arrays, record["methods"])
            except Exception as exc:
                record["preview_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record = {**sample, "error": f"{type(exc).__name__}: {exc}", "methods": {}}
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=json_value) + "\n")
            handle.flush()
        results.append(record)
        print(json.dumps({"done": index, "total": len(samples), "key": sample["key"],
                          "accepted": {m: r.get("accepted", False) for m, r in record["methods"].items()},
                          "seconds": record.get("seconds"), "error": record.get("error")}), flush=True)
    summary = summarize(results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

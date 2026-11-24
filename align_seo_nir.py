"""Create persistent, quality-checked NIR rasters on the S-EO RGB pixel grid."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine


PIPELINE_VERSION = "seo-nir-rgb-grid-v1"


class AlignmentError(ValueError):
    """Registration did not meet the specified quality requirements."""


@dataclass(frozen=True)
class Options:
    max_features: int = 5000
    ratio: float = 0.75
    ransac_threshold: float = 1.5
    min_matches: int = 20
    min_inliers: int = 15
    min_inlier_ratio: float = 0.6
    min_coverage: float = 0.05
    max_holdout_median: float = 0.75
    max_holdout_p90: float = 3.0
    min_holdout_fraction: float = 0.65
    min_valid_fraction: float = 0.1
    seed: int = 42


def resize_coordinate_matrix(source_size_wh, target_size_wh):
    """Map pixel centers using OpenCV's resize convention, including half pixels."""
    sw, sh = source_size_wh
    tw, th = target_size_wh
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = tw / sw, th / sh
    return np.array([[sx, 0, (sx - 1) / 2],
                     [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)


def validate_homography(h, source_size_wh, target_size_wh):
    """Reject singular, reflected, folded, or implausibly scaled image mappings."""
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-10:
        raise AlignmentError("Nonfinite or malformed homography")
    h = h / h[2, 2]
    if abs(np.linalg.det(h)) < 1e-10:
        raise AlignmentError("Singular homography")
    sw, sh = source_size_wh
    tw, th = target_size_wh
    corners = np.array([[-0.5, -0.5], [sw - 0.5, -0.5],
                        [sw - 0.5, sh - 0.5], [-0.5, sh - 0.5]], np.float64)
    denominator = np.c_[corners, np.ones(4)] @ h[2]
    if np.min(denominator) <= 0 or np.max(denominator) / np.min(denominator) > 2:
        raise AlignmentError("Excessive projective distortion")
    mapped = cv2.perspectiveTransform(corners[None], h)[0]
    if not np.isfinite(mapped).all() or not cv2.isContourConvex(mapped.astype(np.float32)):
        raise AlignmentError("Folded image footprint")
    area = cv2.contourArea(mapped.astype(np.float32), oriented=True)
    area_ratio = area / (tw * th)
    if not 0.5 <= area_ratio <= 1.5:
        raise AlignmentError(f"Implausible footprint area ratio: {area_ratio:.3f}")
    if np.any(mapped[:, 0] < -tw) or np.any(mapped[:, 0] > 2 * tw):
        raise AlignmentError("Horizontal displacement exceeds image bounds")
    if np.any(mapped[:, 1] < -th) or np.any(mapped[:, 1] > 2 * th):
        raise AlignmentError("Vertical displacement exceeds image bounds")
    return h


def warp_nir(nir, source_valid, h_native_to_target, target_size_wh):
    """Resample raw DN once; keep only pixels with fully valid bilinear support."""
    nir = np.asarray(nir, dtype=np.float32)
    source_valid = np.asarray(source_valid, dtype=bool)
    if nir.ndim != 2 or source_valid.shape != nir.shape:
        raise ValueError("NIR and validity mask must be matching two-dimensional arrays")
    h = validate_homography(
        h_native_to_target, (nir.shape[1], nir.shape[0]), target_size_wh)
    source_valid = source_valid & np.isfinite(nir)
    values = np.where(source_valid, nir, 0).astype(np.float32)
    kwargs = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned = cv2.warpPerspective(values, h, target_size_wh, **kwargs)
    support = cv2.warpPerspective(source_valid.astype(np.float32), h, target_size_wh, **kwargs)
    valid = (support >= 1 - 1e-6) & np.isfinite(aligned)
    aligned[~valid] = np.nan
    return aligned, valid


def stretch_rgb(rgb):
    """Stretch only the visible matching image, never the NIR measurements."""
    rgb = np.asarray(rgb, dtype=np.float32)
    low, high = np.nanpercentile(rgb, (1, 99), axis=(0, 1))
    return np.uint8(np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1) * 255)


def _mutual_matches(source, target, options, fallback):
    sift = cv2.SIFT_create(nfeatures=options.max_features * (2 if fallback else 1),
                           contrastThreshold=0.01 if fallback else 0.02)
    kp1, desc1 = sift.detectAndCompute(source, None)
    kp2, desc2 = sift.detectAndCompute(target, None)
    if desc1 is None or desc2 is None or min(len(desc1), len(desc2)) < 2:
        raise AlignmentError("Insufficient visible texture")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ratio = min(options.ratio + 0.05, 0.85) if fallback else options.ratio
    forward = [m for pair in matcher.knnMatch(desc1, desc2, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < ratio * n.distance]
    reverse = {(m.trainIdx, m.queryIdx)
               for pair in matcher.knnMatch(desc2, desc1, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < ratio * n.distance}
    matches = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
    if len(matches) < options.min_matches:
        raise AlignmentError(f"Only {len(matches)} mutual matches")
    return (np.float32([kp1[m.queryIdx].pt for m in matches]),
            np.float32([kp2[m.trainIdx].pt for m in matches]))


def _fit_and_check(src, dst, working_size, target_size, options):
    order = np.random.default_rng(options.seed).permutation(len(src))
    split = int(len(order) * 0.8)
    train, holdout = order[:split], order[split:]
    h, inliers = cv2.findHomography(src[train], dst[train], cv2.RANSAC,
                                    options.ransac_threshold, maxIters=10000, confidence=0.999)
    if h is None or inliers is None:
        raise AlignmentError("RANSAC could not estimate a homography")
    h = validate_homography(h, working_size, working_size)
    train_inliers = int(inliers.sum())
    if train_inliers < options.min_inliers or train_inliers / len(train) < options.min_inlier_ratio:
        raise AlignmentError(f"Too few consistent training matches: {train_inliers}/{len(train)}")
    predicted = cv2.perspectiveTransform(src[holdout, None, :], h)[:, 0]
    delta = predicted - dst[holdout]
    errors = np.linalg.norm(delta, axis=1)
    median, p90 = np.percentile(errors, (50, 90))
    fraction = float(np.mean(errors <= options.ransac_threshold))
    if (median > options.max_holdout_median or p90 > options.max_holdout_p90
            or fraction < options.min_holdout_fraction):
        raise AlignmentError(
            f"Holdout mismatch: median={median:.3f}, p90={p90:.3f}, fraction={fraction:.3f}")
    # Refit on every correspondence after checking the held-out correspondences.
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, options.ransac_threshold,
                                    maxIters=10000, confidence=0.999)
    if h is None or inliers is None:
        raise AlignmentError("Final homography fit failed")
    h = validate_homography(h, working_size, working_size)
    refitted = cv2.perspectiveTransform(src[holdout, None, :], h)[:, 0]
    refit_shift = float(np.median(np.linalg.norm(refitted - predicted, axis=1)))
    if refit_shift > 0.5:
        raise AlignmentError(f"Final fit disagrees with the validated fit: {refit_shift:.3f} pixels")
    selected = inliers.ravel().astype(bool)
    count = int(selected.sum())
    if count < options.min_inliers or count / len(src) < options.min_inlier_ratio:
        raise AlignmentError("Final inlier support is insufficient")
    area = working_size[0] * working_size[1]
    coverages = [cv2.contourArea(cv2.convexHull(points[selected])) / area
                 for points in (src, dst)]
    if min(coverages) < options.min_coverage:
        raise AlignmentError(f"Feature coverage too concentrated: {min(coverages):.3f}")
    scale = np.array(target_size, np.float64) / np.array(working_size)
    rgb_errors = np.linalg.norm(delta * scale, axis=1)
    return {
        "H_native_to_target": (resize_coordinate_matrix(working_size, target_size) @ h).tolist(),
        "H_native_to_resized_rgb": h.tolist(),
        "mutual_matches": len(src), "inliers": count,
        "inlier_ratio": count / len(src), "source_coverage": coverages[0],
        "target_coverage": coverages[1], "holdout_count": len(holdout),
        "holdout_median_msi_px": float(median), "holdout_p90_msi_px": float(p90),
        "holdout_median_rgb_px": float(np.median(rgb_errors)),
        "holdout_p90_rgb_px": float(np.percentile(rgb_errors, 90)),
        "holdout_inlier_fraction": fraction,
        "refit_shift_msi_px": refit_shift,
    }


def estimate_registration(msi_rgb_uint8, target_rgb_uint8, options=None):
    options = options or Options()
    sh, sw = msi_rgb_uint8.shape[:2]
    th, tw = target_rgb_uint8.shape[:2]
    small_target = cv2.resize(target_rgb_uint8, (sw, sh), interpolation=cv2.INTER_AREA)
    failures = []
    for fallback in (False, True):
        if fallback:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
            source_gray = clahe.apply(msi_rgb_uint8[:, :, 1])
            target_gray = clahe.apply(small_target[:, :, 1])
        else:
            source_gray = cv2.cvtColor(msi_rgb_uint8, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(small_target, cv2.COLOR_RGB2GRAY)
        try:
            cv2.setRNGSeed(options.seed)
            src, dst = _mutual_matches(source_gray, target_gray, options, fallback)
            result = _fit_and_check(src, dst, (sw, sh), (tw, th), options)
            result["strategy"] = "green_clahe" if fallback else "rgb_gray"
            result["prior_attempt_errors"] = failures
            return result
        except (AlignmentError, cv2.error) as exc:
            failures.append(str(exc))
    raise AlignmentError("; ".join(failures))


def discover_pairs(root):
    root = Path(root)
    pairs = []
    for rgb in sorted((root / "pansharpened_he").rglob("*_rgb.png")):
        relative = rgb.relative_to(root / "pansharpened_he")
        stem = rgb.name.removesuffix("_rgb.png")
        pairs.append({"key": str(relative.with_suffix("")).removesuffix("_rgb").replace("\\", "/"),
                      "rgb": str(rgb),
                      "msi": str(root / "msi" / relative.parent / (stem + "_msi.tif")),
                      "pan": str(root / "crops" / relative.parent / (stem + "_pan.tif")),
                      "relative_parent": str(relative.parent), "stem": stem})
    return pairs


def input_fingerprint(pair):
    return {name: {"size": Path(pair[name]).stat().st_size,
                   "mtime_ns": Path(pair[name]).stat().st_mtime_ns}
            for name in ("rgb", "msi", "pan")}


def output_paths(pair, output):
    parent = Path(output) / pair["relative_parent"]
    return parent / (pair["stem"] + "_nir.tif"), parent / (pair["stem"] + "_valid.png")


def _save_preview(path, source_rgb, target_rgb, h, nir, valid):
    tw, th = target_rgb.shape[1], target_rgb.shape[0]
    size = (320, max(1, round(th * 320 / tw)))
    aligned_rgb = cv2.warpPerspective(source_rgb, h, (tw, th), flags=cv2.INTER_LINEAR)
    target = cv2.resize(target_rgb, size, interpolation=cv2.INTER_AREA)
    aligned = cv2.resize(aligned_rgb, size, interpolation=cv2.INTER_AREA)
    edges1 = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 50, 120)
    edges2 = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), 50, 120)
    overlay = np.stack((edges2, edges1, edges1), axis=-1)
    low, high = np.percentile(nir[valid], (1, 99))
    displayed = np.uint8(np.clip((np.nan_to_num(nir, nan=low) - low) / max(high - low, 1e-6), 0, 1) * 255)
    displayed[~valid] = 0
    gray = cv2.resize(displayed, size, interpolation=cv2.INTER_AREA)
    canvas = Image.new("RGB", (4 * size[0], size[1] + 30), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (target, aligned, overlay, np.repeat(gray[:, :, None], 3, axis=2))
    for index, (panel, label) in enumerate(zip(panels, ("Target RGB", "Aligned MSI visible", "RGB cyan / MSI red edges", "NIR DN (display stretch)"))):
        canvas.paste(Image.fromarray(panel), (index * size[0], 30))
        draw.text((index * size[0] + 5, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def process_pair(pair, output, options=None, preview=False):
    options = options or Options()
    started = time.monotonic()
    record = {"key": pair["key"], "status": "failed", "version": PIPELINE_VERSION}
    try:
        record["inputs"] = input_fingerprint(pair)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(pair["msi"]) as ds:
                if ds.count != 4:
                    raise AlignmentError(f"Expected four raw B/G/R/NIR bands, found {ds.count}")
                source = ds.read()
                source_valid = ds.read_masks(4) > 0
            with rasterio.open(pair["pan"]) as pan:
                pan_size = (pan.width, pan.height)
        with Image.open(pair["rgb"]) as image:
            target_rgb = np.asarray(image.convert("RGB"))
        target_size = (target_rgb.shape[1], target_rgb.shape[0])
        if pan_size != target_size:
            raise AlignmentError(f"PAN/RGB dimensions differ: {pan_size} vs {target_size}")
        source_rgb = stretch_rgb(np.moveaxis(source[[2, 1, 0]], 0, -1))
        registration = estimate_registration(source_rgb, target_rgb, options)
        record.update(registration)
        h = np.array(registration["H_native_to_target"])
        nir, valid = warp_nir(source[3], source_valid, h, target_size)
        record["valid_fraction"] = float(valid.mean())
        record["source_valid_fraction"] = float(source_valid.mean())
        record["coverage_class"] = "full" if record["valid_fraction"] >= 0.95 else "partial"
        if record["valid_fraction"] < options.min_valid_fraction:
            raise AlignmentError(f"Valid image coverage too small: {record['valid_fraction']:.3f}")
        nir_path, valid_path = output_paths(pair, output)
        nir_path.parent.mkdir(parents=True, exist_ok=True)
        if nir_path.exists():
            with rasterio.open(nir_path) as existing:
                if existing.tags().get("PIPELINE_VERSION") != PIPELINE_VERSION:
                    raise AlignmentError("Refusing to overwrite an unrecognized NIR raster")
        temporary_tif = nir_path.with_suffix(".tif.tmp")
        temporary_mask = valid_path.with_suffix(".png.tmp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary_tif, "w", driver="GTiff", width=target_size[0],
                                   height=target_size[1], count=1, dtype="float32", nodata=np.nan,
                                   transform=Affine.identity(), compress="deflate", predictor=3,
                                   zlevel=1, tiled=True, blockxsize=256, blockysize=256) as dst:
                    dst.write(nir, 1)
                    dst.write_mask(valid.astype(np.uint8) * 255)
                    dst.set_band_description(1, "NIR raw DN aligned to RGB pixel grid")
                    dst.update_tags(PIPELINE_VERSION=PIPELINE_VERSION, UNITS="raw_DN",
                                    REFERENCE_RGB=pair["rgb"], SOURCE_MSI=pair["msi"],
                                    H_NATIVE_TO_TARGET=json.dumps(h.tolist()),
                                    GRID="RGB pixels; not an orthorectified map grid")
        encoded, png = cv2.imencode(".png", valid.astype(np.uint8) * 255)
        if not encoded:
            raise IOError("Could not encode validity mask")
        temporary_mask.write_bytes(png.tobytes())
        os.replace(temporary_tif, nir_path)
        os.replace(temporary_mask, valid_path)
        if preview:
            preview_path = Path(output) / "_previews" / pair["relative_parent"] / (pair["stem"] + ".jpg")
            _save_preview(preview_path, source_rgb, target_rgb, h, nir, valid)
        record.update(status="ok", nir=str(nir_path), valid=str(valid_path),
                      target_size_wh=list(target_size), source_size_wh=[source.shape[2], source.shape[1]],
                      output_bytes=nir_path.stat().st_size, valid_bytes=valid_path.stat().st_size,
                      nir_min=float(nir[valid].min()), nir_max=float(nir[valid].max()),
                      units="raw_DN", interpolation="one-pass bilinear", nodata="NaN")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def load_records(path):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[record["key"]] = record
    return records


def reusable(record, pair, output):
    if not record or record.get("status") != "ok" or record.get("version") != PIPELINE_VERSION:
        return False
    nir_path, valid_path = output_paths(pair, output)
    try:
        return (record["inputs"] == input_fingerprint(pair)
                and nir_path.stat().st_size == record["output_bytes"]
                and valid_path.stat().st_size == record["valid_bytes"])
    except (KeyError, OSError):
        return False


def write_csv(path, records):
    fields = ["key", "status", "strategy", "mutual_matches", "inliers", "inlier_ratio",
              "source_coverage", "target_coverage", "holdout_count", "holdout_median_msi_px",
              "holdout_p90_msi_px", "holdout_median_rgb_px", "holdout_p90_rgb_px",
              "valid_fraction", "coverage_class", "nir", "valid", "units", "seconds", "error", "H_native_to_target"]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records.values(), key=lambda item: item["key"]):
            row = dict(record)
            if "H_native_to_target" in row:
                row["H_native_to_target"] = json.dumps(row["H_native_to_target"])
            writer.writerow(row)
    os.replace(temporary, path)


def _worker_init():
    cv2.setNumThreads(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--preview-count", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "nir_aligned_v1").resolve()
    if args.output == args.root or any(args.output == (args.root / name).resolve()
           or (args.root / name).resolve() in args.output.parents
           for name in ("msi", "crops", "pansharpened_he", "shadows_max", "rpcs")):
        parser.error("output must be separate from original data directories")
    options = Options()
    pairs = discover_pairs(args.root)
    if not pairs:
        parser.error("No RGB images were found")
    if args.limit:
        if args.sample:
            pairs = sorted(random.Random(options.seed).sample(pairs, min(args.limit, len(pairs))),
                           key=lambda pair: pair["key"])
        else:
            pairs = pairs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"version": PIPELINE_VERSION, "source_root": str(args.root), "options": asdict(options),
              "algorithm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference": "current pansharpened_he RGB pixel grid", "nir_units": "raw_DN",
              "georeferencing": "pixel alignment only; no unverified RPC/CRS copied"}
    config_path = args.output / "processing_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("Existing output uses a different configuration; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = args.output / "registration.jsonl"
    records = load_records(manifest)
    jobs = [pair for pair in pairs if not reusable(records.get(pair["key"]), pair, args.output)]
    skipped = len(pairs) - len(jobs)
    started = time.monotonic()
    print(json.dumps({"event": "start", "selected": len(pairs), "resume_skipped": skipped,
                      "jobs": len(jobs), "output": str(args.output), "workers": args.workers}), flush=True)
    _worker_init()
    ok = failed = done = 0
    with manifest.open("a", encoding="utf-8") as handle:
        # Isolate an interrupted final JSON line before appending new records.
        handle.write("\n")
        handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {pool.submit(process_pair, pair, args.output, options,
                                   index < args.preview_count): pair for index, pair in enumerate(jobs)}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"key": pair["key"], "status": "failed", "version": PIPELINE_VERSION,
                              "error": f"Worker error: {exc}"}
                records[record["key"]] = record
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                done += 1
                ok += record["status"] == "ok"
                failed += record["status"] != "ok"
                if done % 25 == 0 or done == len(jobs) or record["status"] != "ok":
                    print(json.dumps({"event": "progress", "done": done, "jobs": len(jobs),
                                      "ok": ok, "failed": failed, "last": record["key"],
                                      "error": record.get("error"),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                if done % 250 == 0:
                    write_csv(args.output / "registration.csv", records)
    write_csv(args.output / "registration.csv", records)
    selected_records = [records[pair["key"]] for pair in pairs]
    accepted = [record for record in selected_records if record["status"] == "ok"]
    rejected = [record for record in selected_records if record["status"] != "ok"]
    summary = {"event": "complete", "selected": len(pairs), "accepted": len(accepted),
               "rejected": len(rejected), "resume_skipped": skipped,
               "partial_coverage": sum(r.get("coverage_class") == "partial" for r in accepted),
               "output": str(args.output), "output_bytes": sum(r["output_bytes"] + r["valid_bytes"] for r in accepted),
               "elapsed_seconds": round(time.monotonic() - started, 1),
               "rejected_keys": [record["key"] for record in rejected]}
    if accepted:
        summary["median_holdout_rgb_px"] = float(np.median([r["holdout_median_rgb_px"] for r in accepted]))
        summary["minimum_valid_fraction"] = min(r["valid_fraction"] for r in accepted)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Supplement missing S-EO NIR products using verified PAN registration."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import csv

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from align_seo_nir import (discover_pairs, input_fingerprint, load_records,
                          resize_coordinate_matrix, stretch_rgb, validate_homography, warp_nir)
from nir_registration_metrics import (fit_and_validate_spatial, registration_support_mask,
                                      spatial_holdout_split, _balanced_subset)
from nir_registration_pan import assess_pan_rgb_grid
from nir_registration_benchmark.pan_fullres_probe import match_full_resolution


VERSION = "seo-nir-pan-recovery-v2"
HERE = Path(__file__).resolve().parent
PREFIX = "pan_recovery"
_LOFTR = None


@dataclass(frozen=True)
class RecoveryOptions:
    min_valid_fraction: float = 0.1
    use_loftr: bool = True
    weights: str = str(HERE / "nir_registration_benchmark" / "loftr_outdoor.ckpt")


def validate_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output:
        raise ValueError("Output must be a dedicated NIR directory")
    for name in ("msi", "crops", "pansharpened_he", "shadows_max", "shadows_min", "rpcs",
                 "rdsm_max_compressed", "rdsm_min_compressed"):
        source = root / name
        if output == source or source in output.parents:
            raise ValueError("Output cannot be an original input directory")


def product_paths(pair, output, grid):
    if grid not in ("pan", "rgb"):
        raise ValueError("Unknown reference grid")
    parent = Path(output) / pair["relative_parent"]
    return parent / (pair["stem"] + "_nir.tif"), parent / (pair["stem"] + "_valid.png")


def _display_pan(pan):
    finite = np.isfinite(pan)
    low, high = np.percentile(pan[finite], (1, 99))
    return np.uint8(np.clip((pan.astype(np.float32) - low) / max(high - low, 1), 0, 1) * 255)


def _matcher(options):
    global _LOFTR
    if _LOFTR is None:
        from compare_seo_nir_registration import LoFTRMatcher
        _LOFTR = LoFTRMatcher(Path(options.weights))
    return _LOFTR


def estimate_pan_registration(source_rgb, pan_uint8, options, matcher=None):
    """Try native-PAN SIFT variants, then learned matching against PAN only."""
    source_size = (source_rgb.shape[1], source_rgb.shape[0])
    target_size = (pan_uint8.shape[1], pan_uint8.shape[0])
    attempts = []
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    for strategy in ("native_pan_sift_gray", "native_pan_sift_green_clahe", "pan_loftr"):
        if strategy == "pan_loftr" and not options.use_loftr:
            continue
        try:
            if strategy == "native_pan_sift_gray":
                src, dst, stats = match_full_resolution(source_rgb, pan_uint8, "sift5000_mutual_ratio")
                candidates = [(src, dst, None, stats)]
            elif strategy == "native_pan_sift_green_clahe":
                green = np.repeat(clahe.apply(source_rgb[:, :, 1])[:, :, None], 3, axis=2)
                src, dst, stats = match_full_resolution(green, clahe.apply(pan_uint8), "sift5000_mutual_ratio")
                candidates = [(src, dst, None, stats)]
            else:
                matcher = matcher or _matcher(options)
                src, dst, scores = matcher.match(source_rgb, pan_uint8)
                candidates = []
                for threshold in (0.5, 0.2):
                    keep = scores >= threshold
                    candidates.append((src[keep], dst[keep], scores[keep],
                                       {"confidence": threshold, "match_count": int(keep.sum())}))
            for src, dst, scores, stats in candidates:
                fitted = fit_and_validate_spatial(src, dst, source_size, target_size, confidence=scores)
                attempts.append({"strategy": strategy, **stats, "accepted": fitted["accepted"],
                                 "reason": fitted.get("reason"), "metrics": fitted["metrics"]})
                if fitted["accepted"]:
                    fraction = float(registration_support_mask(np.ones(source_rgb.shape[:2], bool),
                                     np.asarray(fitted["H_native_to_target"]), target_size).mean())
                    if fraction < options.min_valid_fraction:
                        attempts[-1].update(accepted=False, reason="Too little common PAN coverage")
                        continue
                    return {**fitted, "strategy": strategy, "attempts": attempts,
                            "_source_points": src, "_pan_points": dst}
        except Exception as exc:
            attempts.append({"strategy": strategy, "accepted": False,
                             "execution_error": f"{type(exc).__name__}: {exc}"})
            gc.collect()
    return {"accepted": False, "H_native_to_target": None, "strategy": None,
            "metrics": {}, "attempts": attempts, "reason": "No PAN candidate passed spatial validation"}


def estimate_rgb_bridge(pan_uint8, target_rgb, source_size_wh, options):
    grid = assess_pan_rgb_grid(pan_uint8, target_rgb)
    pan_size = (pan_uint8.shape[1], pan_uint8.shape[0])
    target_size = (target_rgb.shape[1], target_rgb.shape[0])
    pan_to_work = resize_coordinate_matrix(pan_size, source_size_wh)
    attempts = []
    # Revalidate every candidate with the same spatial protocol as MSI to PAN.
    for attempt in grid["attempts"]:
        src = np.asarray(attempt["points_src_pan"], np.float64).reshape(-1, 2)
        dst = np.asarray(attempt["points_dst_rgb"], np.float64).reshape(-1, 2)
        work = cv2.perspectiveTransform(src[:, None], pan_to_work)[:, 0] if len(src) else src
        checked = fit_and_validate_spatial(work, dst, source_size_wh, target_size)
        attempts.append({"strategy": attempt["strategy"], "accepted": checked["accepted"],
                         "reason": checked.get("reason"), "metrics": checked["metrics"]})
        if checked["accepted"]:
            h = np.asarray(checked["H_native_to_target"]) @ pan_to_work
            return {"accepted": True, "H_pan_to_rgb": h.tolist(), "metrics": checked["metrics"],
                    "reason": None, "attempts": attempts}
    return {"accepted": False, "H_pan_to_rgb": None, "metrics": {}, "attempts": attempts,
            "reason": "PAN to existing RGB could not be spatially verified"}


def check_composition(source_points, pan_points, h, h_pan_to_rgb, source_size, target_size):
    src = np.asarray(source_points, np.float64).reshape(-1, 2)
    target = cv2.perspectiveTransform(np.asarray(pan_points, np.float64)[:, None], h_pan_to_rgb)[:, 0]
    sw, sh = source_size
    tw, th = target_size
    keep = ((src[:, 0] >= 0) & (src[:, 0] < sw) & (src[:, 1] >= 0) & (src[:, 1] < sh)
            & (target[:, 0] >= 0) & (target[:, 0] < tw) & (target[:, 1] >= 0) & (target[:, 1] < th))
    src, target = src[keep], target[keep]
    fitted = fit_and_validate_spatial(src, target, source_size, target_size)
    if not fitted["accepted"]:
        return {"accepted": False, "reason": fitted["reason"], "metrics": fitted["metrics"]}
    selected = _balanced_subset(src, source_size, (4, 4), 3000, None, 42)
    src, target = src[selected], target[selected]
    held = spatial_holdout_split(src, target, source_size)["test_indices"]
    work = resize_coordinate_matrix(target_size, source_size)
    predicted = cv2.perspectiveTransform(src[held, None], work @ h)[:, 0]
    expected = cv2.perspectiveTransform(target[held, None], work)[:, 0]
    errors = np.linalg.norm(predicted - expected, axis=1)
    median, p90 = np.percentile(errors, (50, 90))
    fraction = float(np.mean(errors <= 1.5))
    return {"accepted": bool(median <= 0.75 and p90 <= 3 and fraction >= 0.65),
            "reason": None if median <= 0.75 and p90 <= 3 and fraction >= 0.65 else "Composition holdout error too large",
            "metrics": {"holdout_median_msi_px": float(median), "holdout_p90_msi_px": float(p90),
                        "holdout_inlier_fraction": fraction}}


def save_product(raw_nir, source_valid, h, target_size_wh, pair, output, grid, config_hash):
    nir_path, valid_path = product_paths(pair, output, grid)
    if nir_path.exists():
        with rasterio.open(nir_path) as existing:
            tags = existing.tags()
            if tags.get("PIPELINE_VERSION") != VERSION or tags.get("RECOVERY_CONFIG") != config_hash:
                raise FileExistsError("Refusing to overwrite an existing NIR product from another run")
    elif valid_path.exists():
        raise FileExistsError("Refusing to overwrite an unpaired existing validity mask")
    aligned, valid = warp_nir(raw_nir, source_valid, h, target_size_wh)
    if valid.mean() < 0.1:
        raise ValueError("Less than ten percent valid NIR coverage")
    nir_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tif = nir_path.with_name(nir_path.name + ".pan-recovery.tmp")
    temporary_png = valid_path.with_name(valid_path.name + ".pan-recovery.tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=32 * 1024 * 1024):
            with rasterio.open(temporary_tif, "w", driver="GTiff", width=target_size_wh[0],
                               height=target_size_wh[1], count=1, dtype="float32", nodata=np.nan,
                               transform=Affine.identity(), tiled=True, blockxsize=256, blockysize=256,
                               compress="deflate", zlevel=1, predictor=3) as destination:
                destination.write(aligned, 1)
                destination.write_mask(valid.astype(np.uint8) * 255)
                destination.set_band_description(1, "NIR raw DN, PAN-based registration")
                destination.update_tags(PIPELINE_VERSION=VERSION, RECOVERY_CONFIG=config_hash,
                    SOURCE_MSI=pair["msi"], REFERENCE_PAN=pair["pan"], REFERENCE_RGB=pair["rgb"],
                    REFERENCE_GRID=grid.upper(), CURRENT_RGB_READY=str(grid == "rgb").lower(),
                    UNITS="raw_DN", H_NATIVE_TO_TARGET=json.dumps(np.asarray(h).tolist()),
                    GRID=grid.upper() + " pixel grid; no unverified map georeferencing copied")
    encoded, png = cv2.imencode(".png", valid.astype(np.uint8) * 255)
    if not encoded:
        raise RuntimeError("Validity PNG could not be encoded")
    temporary_png.write_bytes(png.tobytes())
    os.replace(temporary_tif, nir_path)
    os.replace(temporary_png, valid_path)
    values = aligned[valid]
    return {"nir": str(nir_path), "valid": str(valid_path), "reference_grid": grid.upper(),
            "current_rgb_ready": grid == "rgb", "target_size_wh": list(target_size_wh),
            "valid_fraction": float(valid.mean()), "nir_min": float(values.min()),
            "nir_max": float(values.max()), "output_bytes": nir_path.stat().st_size,
            "valid_bytes": valid_path.stat().st_size, "units": "raw_DN", "nodata": "NaN"}


def process_pair(pair, output, options=None, matcher=None, preview=False, config_hash="test"):
    options = options or RecoveryOptions()
    started = time.monotonic()
    cv2.setNumThreads(1)
    record = {"key": pair["key"], "status": "error", "pan_accepted": False, "rgb_ready": False,
              "reference_grid": None, "version": VERSION, "config_hash": config_hash}
    try:
        record["inputs"] = input_fingerprint(pair)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
                with rasterio.open(pair["msi"]) as dataset:
                    if dataset.count != 4:
                        raise ValueError("Expected raw B/G/R/NIR MSI")
                    raw = dataset.read()
                    source_valid = dataset.read_masks(4) > 0
                with rasterio.open(pair["pan"]) as dataset:
                    pan = _display_pan(dataset.read(1))
        with Image.open(pair["rgb"]) as image:
            target = np.asarray(image.convert("RGB"))
        visible = stretch_rgb(np.moveaxis(raw[[2, 1, 0]], 0, -1))
        source_size, pan_size = (raw.shape[2], raw.shape[1]), (pan.shape[1], pan.shape[0])
        target_size = (target.shape[1], target.shape[0])
        pan_fit = estimate_pan_registration(visible, pan, options, matcher)
        source_points = pan_fit.pop("_source_points", None)
        pan_points = pan_fit.pop("_pan_points", None)
        record["pan_registration"] = pan_fit
        if not pan_fit["accepted"]:
            retry_needed = any(attempt.get("execution_error") for attempt in pan_fit.get("attempts", []))
            record.update(status="error" if retry_needed else "rejected", reason=pan_fit.get("reason"))
            return record
        h_pan = validate_homography(np.asarray(pan_fit["H_native_to_target"]), source_size, pan_size)
        support = registration_support_mask(source_valid, h_pan, pan_size)
        if support.mean() < options.min_valid_fraction:
            record.update(status="rejected", reason="Actual NIR/PAN valid coverage is too small")
            return record
        record["pan_accepted"] = True
        record["pan_valid_fraction"] = float(support.mean())
        record["H_native_to_pan"] = h_pan.tolist()
        try:
            bridge = estimate_rgb_bridge(pan, target, source_size, options)
        except Exception as exc:
            bridge = {"accepted": False, "reason": f"{type(exc).__name__}: {exc}", "execution_error": True}
        record["rgb_bridge"] = bridge
        h, size, grid = h_pan, pan_size, "pan"
        if bridge["accepted"]:
            try:
                h_grid = np.asarray(bridge["H_pan_to_rgb"], np.float64)
                composed = validate_homography(h_grid @ h_pan, source_size, target_size)
                check = (check_composition(source_points, pan_points, composed, h_grid, source_size, target_size)
                         if source_points is not None else {"accepted": True, "reason": None})
                record["composition_validation"] = check
                rgb_support = registration_support_mask(source_valid, composed, target_size)
                if check["accepted"] and rgb_support.mean() >= options.min_valid_fraction:
                    h, size, grid = composed, target_size, "rgb"
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                record["composition_validation"] = {"accepted": False, "reason": str(exc)}
        product = save_product(raw[3], source_valid, h, size, pair, output, grid, config_hash)
        record.update(status="rgb_ready" if grid == "rgb" else "pan_only", rgb_ready=grid == "rgb",
                      reference_grid=grid.upper(), product=product, H_native_to_target=h.tolist())
        if preview:
            try:
                aligned, valid = warp_nir(raw[3], source_valid, h, size)
                low, high = np.percentile(aligned[valid], (1, 99))
                shown = np.uint8(np.clip((np.nan_to_num(aligned, nan=low) - low) / max(high - low, 1), 0, 1) * 255)
                shown[~valid] = 0
                path = Path(output) / "_pan_recovery_previews" / pair["relative_parent"] / (pair["stem"] + "_nir.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(shown).save(path)
                record["preview"] = str(path)
            except Exception as exc:
                record["preview_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)
        gc.collect()
    return record


def reusable(record, pair, output, config_hash):
    if not record or record.get("config_hash") != config_hash or record.get("status") == "error":
        return False
    try:
        if record["inputs"] != input_fingerprint(pair):
            return False
        if record.get("rgb_bridge", {}).get("execution_error"):
            return False
        if record["status"] == "rejected":
            return True
        product = record["product"]
        expected = product_paths(pair, output, product["reference_grid"].lower())
        return (Path(product["nir"]) == expected[0] and Path(product["valid"]) == expected[1]
                and expected[0].stat().st_size == product["output_bytes"]
                and expected[1].stat().st_size == product["valid_bytes"])
    except (KeyError, OSError):
        return False


def select_remaining_pairs(root, originals, keys=None):
    missing = {key for key, record in originals.items() if record["status"] != "ok"}
    if keys is not None:
        requested = set(keys)
        if not requested <= missing:
            raise ValueError("Only previously failed keys can be supplemented")
        missing = requested
    pairs = [pair for pair in discover_pairs(root) if pair["key"] in missing]
    if {pair["key"] for pair in pairs} != missing:
        raise ValueError("Some selected keys have no current RGB pair")
    return pairs


def write_tables(output, originals, recovered, selected_keys):
    fields = ["key", "status", "reference_grid", "current_rgb_ready", "nir", "valid", "valid_fraction",
              "strategy", "holdout_median_msi_px", "holdout_p90_msi_px", "reason"]
    all_rows, recovery_rows = [], []
    for key, record in sorted(originals.items()):
        if record["status"] == "ok":
            all_rows.append({"key": key, "status": "original_rgb_ready", "reference_grid": "RGB",
                             "current_rgb_ready": True, **{k: record.get(k) for k in ("nir", "valid", "valid_fraction")}})
        elif key in recovered:
            new = recovered[key]
            fit = new.get("pan_registration", {})
            row = {"key": key, "status": new["status"], "reference_grid": new.get("reference_grid"),
                   "current_rgb_ready": new.get("rgb_ready", False), **new.get("product", {}),
                   "strategy": fit.get("strategy"), "reason": new.get("reason", fit.get("reason"))}
            row.update({k: fit.get("metrics", {}).get(k) for k in ("holdout_median_msi_px", "holdout_p90_msi_px")})
            all_rows.append(row)
            if key in selected_keys:
                recovery_rows.append(row)
        else:
            all_rows.append({"key": key, "status": "awaiting_pan_recovery", "current_rgb_ready": False,
                             "reason": record.get("error")})
    for name, rows in ((PREFIX + "_registration.csv", recovery_rows), ("registration_all.csv", all_rows),
                       ("rgb_training_manifest.csv", [r for r in all_rows if r.get("current_rgb_ready")])):
        destination = Path(output) / name
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--output", type=Path, default=Path("E:/Dataset/SEO/nir_aligned_v1"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keys", help="Optional comma-separated subset of failed keys for a pilot")
    parser.add_argument("--no-loftr", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    args = parser.parse_args()
    validate_output(args.root, args.output)
    original_path = args.root / "nir_aligned_v1" / "registration.jsonl"
    originals = load_records(original_path)
    pairs = select_remaining_pairs(args.root, originals, args.keys.split(",") if args.keys else None)
    missing = {pair["key"] for pair in pairs}
    options = RecoveryOptions(use_loftr=not args.no_loftr)
    source_files = ("recover_seo_nir_pan.py", "align_seo_nir.py", "nir_registration_metrics.py", "nir_registration_pan.py",
                    "compare_seo_nir_registration.py", "nir_registration_benchmark/pan_fullres_probe.py")
    config = {"version": VERSION, "source_root": str(args.root.resolve()), "output": str(args.output.resolve()),
              "selected_keys": sorted(missing), "options": asdict(options),
              "original_manifest_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in source_files}}
    if options.use_loftr:
        config["weights_sha256"] = hashlib.sha256(Path(options.weights).read_bytes()).hexdigest()
    serialized = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(serialized.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / (PREFIX + "_config.json")
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing recovery configuration differs; do not mix recovery runs")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = args.output / (PREFIX + "_registration.jsonl")
    records = load_records(manifest)
    jobs = [pair for pair in pairs if not reusable(records.get(pair["key"]), pair, args.output, config_hash)
            or (args.retry_rejected and records.get(pair["key"], {}).get("status") == "rejected")]
    started = time.monotonic()
    options_dict = options
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_pair, pair, args.output, options_dict, None, True, config_hash): pair for pair in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"key": pair["key"], "status": "error", "reason": f"{type(exc).__name__}: {exc}",
                              "config_hash": config_hash}
                records[record["key"]] = record
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                selected = [records[k] for k in missing if k in records]
                if index % 10 == 0 or index == len(jobs):
                    progress = {"done": len(selected), "total": len(pairs),
                                "saved": sum(r["status"] in ("rgb_ready", "pan_only") for r in selected),
                                "rgb_ready": sum(r["status"] == "rgb_ready" for r in selected),
                                "pan_only": sum(r["status"] == "pan_only" for r in selected),
                                "rejected": sum(r["status"] == "rejected" for r in selected),
                                "errors": sum(r["status"] == "error" for r in selected),
                                "elapsed_seconds": round(time.monotonic() - started, 1)}
                    print(json.dumps(progress), flush=True)
                    write_tables(args.output, originals, records, missing)
    selected = [records[pair["key"]] for pair in pairs]
    summary = {"selected": len(pairs), "attempted_this_run": len(jobs), "output": str(args.output),
               "saved": sum(r["status"] in ("rgb_ready", "pan_only") for r in selected),
               "rgb_ready": sum(r["status"] == "rgb_ready" for r in selected),
               "pan_only": sum(r["status"] == "pan_only" for r in selected),
               "rejected": sum(r["status"] == "rejected" for r in selected),
               "errors": sum(r["status"] == "error" for r in selected),
               "elapsed_seconds": round(time.monotonic() - started, 1)}
    write_tables(args.output, originals, records, missing)
    (args.output / (PREFIX + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

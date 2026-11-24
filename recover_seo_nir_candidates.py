"""Save explicitly unverified PAN-based candidates for remaining S-EO NIR keys."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import as_completed
import csv
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
from multiprocessing import freeze_support
from pathlib import Path
import re
import time
import warnings

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from align_seo_nir import (discover_pairs, input_fingerprint, load_records,
                          stretch_rgb, validate_homography, warp_nir)
from nir_registration_benchmark.pan_fullres_probe import match_full_resolution
from nir_registration_metrics import registration_support_mask
from nir_registration_relaxed import RelaxedOptions, fit_relaxed
from recover_seo_nir_pan import (_display_pan, _matcher, estimate_rgb_bridge,
                                validate_output)
from recover_seo_nir_pan_isolated import IsolatedImageExecutor


HERE = Path(__file__).resolve().parent
VERSION = "seo-nir-pan-candidate-v3"
PREFIX = "pan_candidate"
SAVED = {"candidate_rgb", "candidate_pan"}
QUALITY = "UNVERIFIED_CANDIDATE"
PROTECTED_LEDGERS = ("registration.jsonl", "registration.csv", "processing_config.json",
                     "summary.json", "pan_recovery_config.json", "pan_recovery_registration.jsonl",
                     "pan_recovery_registration.csv", "pan_recovery_summary.json",
                     "registration_all.csv", "rgb_training_manifest.csv")


@dataclass(frozen=True)
class CandidateOptions:
    fit: RelaxedOptions = field(default_factory=RelaxedOptions)
    min_valid_fraction: float = 0.1
    use_loftr: bool = True
    weights: str = str(HERE / "nir_registration_benchmark" / "loftr_outdoor.ckpt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".candidate.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def product_paths(pair, output):
    parent = Path(output) / pair["relative_parent"]
    return parent / (pair["stem"] + "_nir.tif"), parent / (pair["stem"] + "_valid.png")


def resource_error_reasons(value):
    if isinstance(value, dict):
        found = []
        for key, child in value.items():
            if key in ("reason", "execution_error") and isinstance(child, str):
                if any(token in child.lower() for token in
                       ("outofmemory", "out of memory", "insufficient memory", "unable to allocate",
                        "bad allocation", "memoryerror", "std::bad_alloc")):
                    found.append(child)
            else:
                found.extend(resource_error_reasons(child))
        return found
    if isinstance(value, list):
        return [reason for child in value for reason in resource_error_reasons(child)]
    return []


def select_remaining_pairs(root, originals, strict_recovered, keys=None):
    missing = {key for key, record in strict_recovered.items()
               if record.get("status") == "rejected" and key in originals
               and originals[key].get("status") != "ok"}
    if keys is not None:
        requested = set(keys)
        if not requested <= missing:
            raise ValueError("Candidate recovery may select only strictly rejected, originally failed keys")
        missing = requested
    pairs = [pair for pair in discover_pairs(root) if pair["key"] in missing]
    if {pair["key"] for pair in pairs} != missing:
        raise ValueError("Some selected keys have no current RGB/MSI/PAN pair")
    return pairs


def estimate_candidate_registration(source_rgb, pan, options, matcher=None):
    source_size = (source_rgb.shape[1], source_rgb.shape[0])
    pan_size = (pan.shape[1], pan.shape[0])
    attempts = []
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    for strategy in ("native_pan_sift_gray", "native_pan_sift_green_clahe", "pan_loftr"):
        if strategy == "pan_loftr" and not options.use_loftr:
            continue
        try:
            if strategy == "native_pan_sift_gray":
                src, dst, stats = match_full_resolution(source_rgb, pan, "sift5000_mutual_ratio")
                candidates = [(src, dst, None, stats)]
            elif strategy == "native_pan_sift_green_clahe":
                green = np.repeat(clahe.apply(source_rgb[:, :, 1])[:, :, None], 3, axis=2)
                src, dst, stats = match_full_resolution(green, clahe.apply(pan), "sift5000_mutual_ratio")
                candidates = [(src, dst, None, stats)]
            else:
                matcher = matcher or _matcher(options)
                src, dst, scores = matcher.match(source_rgb, pan)
                candidates = []
                for threshold in (0.5, 0.2):
                    keep = scores >= threshold
                    candidates.append((src[keep], dst[keep], scores[keep],
                                       {"confidence": threshold, "match_count": int(keep.sum())}))
            for src, dst, scores, stats in candidates:
                fitted = fit_relaxed(src, dst, source_size, pan_size,
                                     confidence=scores, options=options.fit)
                attempt = {"strategy": strategy, **stats, **fitted}
                attempts.append(attempt)
                if not fitted["accepted"]:
                    continue
                support = registration_support_mask(np.ones(source_rgb.shape[:2], bool),
                                                   np.asarray(fitted["H_native_to_target"]), pan_size)
                fraction = float(support.mean())
                attempt["canvas_valid_fraction"] = fraction
                if fraction < options.min_valid_fraction:
                    attempt.update(accepted=False, reason="Less than ten percent common PAN coverage")
                    continue
                return {**fitted, "strategy": strategy, "attempts": attempts}
        except Exception as exc:
            attempts.append({"strategy": strategy, "accepted": False,
                             "execution_error": f"{type(exc).__name__}: {exc}"})
            gc.collect()
    return {"accepted": False, "H_native_to_target": None, "metrics": {},
            "strategy": None, "attempts": attempts,
            "reason": "No candidate met basic geometry and relaxed support requirements"}


def save_product(raw_nir, source_valid, h, target_size_wh, pair, output, grid,
                 config_hash, bridge_verified=False):
    if grid not in ("pan", "rgb"):
        raise ValueError("Unknown candidate reference grid")
    nir_path, valid_path = product_paths(pair, output)
    if nir_path.exists():
        with rasterio.open(nir_path) as existing:
            tags = existing.tags()
            if (tags.get("PIPELINE_VERSION") != VERSION or tags.get("RECOVERY_CONFIG") != config_hash
                    or tags.get("SOURCE_MSI") != pair["msi"]):
                raise FileExistsError("Refusing to overwrite an existing original or foreign NIR product")
    elif valid_path.exists():
        raise FileExistsError("Refusing to overwrite an orphan existing validity mask")
    aligned, valid = warp_nir(raw_nir, source_valid, h, target_size_wh)
    if float(valid.mean()) < 0.1:
        raise ValueError("Less than ten percent valid NIR coverage")
    nir_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tif = nir_path.with_name(nir_path.name + ".candidate.tmp")
    temporary_png = valid_path.with_name(valid_path.name + ".candidate.tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=32 * 1024 * 1024):
            with rasterio.open(temporary_tif, "w", driver="GTiff", width=target_size_wh[0],
                               height=target_size_wh[1], count=1, dtype="float32", nodata=np.nan,
                               transform=Affine.identity(), tiled=True, blockxsize=256, blockysize=256,
                               compress="deflate", zlevel=1, predictor=3) as destination:
                destination.write(aligned, 1)
                destination.write_mask(valid.astype(np.uint8) * 255)
                destination.set_band_description(1, "NIR raw DN, unverified registration candidate")
                destination.update_tags(PIPELINE_VERSION=VERSION, RECOVERY_CONFIG=config_hash,
                    REGISTRATION_STATUS=QUALITY, REGISTRATION_VERIFIED="false", CURRENT_RGB_READY="false",
                    RGB_GRID_VERIFIED=str(bool(bridge_verified)).lower(), REFERENCE_GRID=grid.upper(),
                    SOURCE_MSI=pair["msi"], REFERENCE_PAN=pair["pan"], REFERENCE_RGB=pair["rgb"],
                    STATISTICAL_VALIDATION="relaxed_support_no_spatial_holdout_gate", UNITS="raw_DN",
                    H_NATIVE_TO_TARGET=json.dumps(np.asarray(h).tolist()),
                    GRID=grid.upper() + " pixel grid; candidate geometry is not independently verified")
    encoded, png = cv2.imencode(".png", valid.astype(np.uint8) * 255)
    if not encoded:
        raise RuntimeError("Could not encode candidate validity mask")
    temporary_png.write_bytes(png.tobytes())
    os.replace(temporary_tif, nir_path)
    os.replace(temporary_png, valid_path)
    values = aligned[valid]
    return {"nir": str(nir_path), "valid": str(valid_path), "reference_grid": grid.upper(),
            "quality_status": QUALITY, "registration_verified": False, "current_rgb_ready": False,
            "rgb_grid_verified": bool(bridge_verified), "target_size_wh": list(target_size_wh),
            "valid_fraction": float(valid.mean()), "nir_min": float(values.min()),
            "nir_max": float(values.max()), "output_bytes": nir_path.stat().st_size,
            "valid_bytes": valid_path.stat().st_size, "units": "raw_DN", "nodata": "NaN"}


def process_pair(pair, output, options=None, matcher=None, preview=False, config_hash="test"):
    options = options or CandidateOptions()
    started = time.monotonic()
    cv2.setNumThreads(1)
    record = {"key": pair["key"], "status": "error", "version": VERSION,
              "config_hash": config_hash, "quality_status": QUALITY,
              "registration_verified": False, "current_rgb_ready": False}
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
        source_size = (raw.shape[2], raw.shape[1])
        pan_size, target_size = (pan.shape[1], pan.shape[0]), (target.shape[1], target.shape[0])
        fitted = estimate_candidate_registration(visible, pan, options, matcher)
        record["pan_registration"] = fitted
        if not fitted["accepted"]:
            retry = any(attempt.get("execution_error") for attempt in fitted.get("attempts", []))
            record.update(status="error" if retry else "rejected", reason=fitted.get("reason"))
            return record
        h_pan = validate_homography(np.asarray(fitted["H_native_to_target"]), source_size, pan_size)
        support = registration_support_mask(source_valid, h_pan, pan_size)
        if float(support.mean()) < options.min_valid_fraction:
            record.update(status="rejected", reason="Actual NIR/PAN valid coverage is below ten percent")
            return record
        record.update(H_native_to_pan=h_pan.tolist(), pan_valid_fraction=float(support.mean()))
        try:
            bridge = estimate_rgb_bridge(pan, target, source_size, options)
            resource_errors = resource_error_reasons(bridge)
            if resource_errors:
                bridge["execution_error"] = "; ".join(dict.fromkeys(resource_errors))
        except Exception as exc:
            bridge = {"accepted": False, "execution_error": f"{type(exc).__name__}: {exc}",
                      "reason": "PAN to current RGB bridge execution failed"}
        record["rgb_bridge"] = bridge
        h, size, grid, bridge_verified = h_pan, pan_size, "pan", False
        if bridge.get("accepted"):
            try:
                composed = validate_homography(np.asarray(bridge["H_pan_to_rgb"]) @ h_pan,
                                               source_size, target_size)
                rgb_support = registration_support_mask(source_valid, composed, target_size)
                if float(rgb_support.mean()) >= options.min_valid_fraction:
                    h, size, grid, bridge_verified = composed, target_size, "rgb", True
                else:
                    record["composition_reason"] = "Insufficient actual NIR support on RGB canvas"
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                if isinstance(exc, cv2.error) and (getattr(exc, "code", None) == -4
                        or "memory" in str(exc).lower()):
                    raise
                record["composition_reason"] = f"{type(exc).__name__}: {exc}"
        product = save_product(raw[3], source_valid, h, size, pair, output, grid,
                               config_hash, bridge_verified=bridge_verified)
        record.update(status="candidate_rgb" if grid == "rgb" else "candidate_pan",
                      reference_grid=grid.upper(), product=product, H_native_to_target=h.tolist())
        if preview:
            try:
                aligned, valid = warp_nir(raw[3], source_valid, h, size)
                low, high = np.percentile(aligned[valid], (1, 99))
                shown = np.uint8(np.clip((np.nan_to_num(aligned, nan=low) - low)
                                        / max(high - low, 1), 0, 1) * 255)
                shown[~valid] = 0
                path = Path(output) / "_pan_candidate_previews" / pair["relative_parent"] / (pair["stem"] + "_nir.png")
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


def execution_events(value):
    if isinstance(value, dict):
        return ([value["execution_error"]] if value.get("execution_error") else []) + [
            item for key, child in value.items() if key != "execution_error" for item in execution_events(child)]
    if isinstance(value, list):
        return [item for child in value for item in execution_events(child)]
    return []


def reusable(record, pair, output, config_hash):
    if not record or record.get("config_hash") != config_hash or record.get("status") == "error":
        return False
    try:
        if record.get("inputs") != input_fingerprint(pair) or execution_events(record) or record.get("preview_error"):
            return False
        if record["status"] == "rejected":
            return True
        if record["status"] not in SAVED or record.get("registration_verified") is not False:
            return False
        product = record["product"]
        paths = product_paths(pair, output)
        return (Path(product["nir"]) == paths[0] and Path(product["valid"]) == paths[1]
                and paths[0].stat().st_size == product["output_bytes"]
                and paths[1].stat().st_size == product["valid_bytes"])
    except (KeyError, OSError):
        return False


def capture_baseline(output, selected_keys):
    output = Path(output)
    ledgers = {name: sha256(output / name) for name in PROTECTED_LEDGERS if (output / name).is_file()}
    products = {}
    if output.exists():
        for folder in output.iterdir():
            if not folder.is_dir() or not re.fullmatch(r"(?:JAX|OMA|UCSD)_\d+", folder.name):
                continue
            for suffix in ("_nir.tif", "_valid.png"):
                for path in folder.glob("*" + suffix):
                    key = folder.name + "/" + path.name.removesuffix(suffix)
                    if key in selected_keys:
                        continue
                    info = path.stat()
                    products[path.relative_to(output).as_posix()] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    return {"protected_ledgers_sha256": ledgers, "protected_products": products,
            "protected_nir_count": sum(path.endswith("_nir.tif") for path in products),
            "protected_valid_count": sum(path.endswith("_valid.png") for path in products)}


def check_baseline(output, baseline):
    output = Path(output)
    issues = []
    for name, digest in baseline["protected_ledgers_sha256"].items():
        if not (output / name).is_file() or sha256(output / name) != digest:
            issues.append("Historical ledger changed: " + name)
    for name, info in baseline["protected_products"].items():
        path = output / name
        if not path.is_file():
            issues.append("Historical product missing: " + name)
        else:
            current = path.stat()
            if current.st_size != info["size"] or current.st_mtime_ns != info["mtime_ns"]:
                issues.append("Historical product metadata changed: " + name)
    return issues


def write_tables(output, originals, strict, candidates, selected):
    fields = ["key", "status", "quality_status", "registration_verified", "reference_grid",
              "current_rgb_ready", "rgb_grid_verified", "nir", "valid", "valid_fraction",
              "strategy", "model", "inliers", "inlier_ratio", "training_inlier_median_msi_px", "reason"]
    all_rows, candidate_rows = [], []
    for key, old in sorted(originals.items()):
        if key in candidates:
            record = candidates[key]
            fit = record.get("pan_registration", {})
            row = {"key": key, "status": record["status"], "quality_status": QUALITY,
                   "registration_verified": False, "current_rgb_ready": False,
                   "reason": record.get("reason", fit.get("reason")), **record.get("product", {}),
                   "strategy": fit.get("strategy"), "model": fit.get("model")}
            row.update({name: fit.get("metrics", {}).get(name) for name in
                        ("inliers", "inlier_ratio", "training_inlier_median_msi_px")})
            candidate_rows.append(row)
        elif old.get("status") == "ok":
            row = {"key": key, "status": "original_rgb_ready", "quality_status": "ORIGINAL_ACCEPTED",
                   "registration_verified": True, "current_rgb_ready": True, "reference_grid": "RGB",
                   **{name: old.get(name) for name in ("nir", "valid", "valid_fraction")}}
        elif key in strict:
            record = strict[key]
            accepted = record["status"] in ("rgb_ready", "pan_only")
            row = {"key": key, "status": record["status"],
                   "quality_status": "STRICT_ACCEPTED" if accepted else "STRICT_REJECTED",
                   "registration_verified": accepted, "current_rgb_ready": record["status"] == "rgb_ready",
                   "reference_grid": record.get("reference_grid"), "reason": record.get("reason"),
                   **record.get("product", {})}
        else:
            row = {"key": key, "status": "awaiting_recovery", "registration_verified": False,
                   "current_rgb_ready": False}
        all_rows.append(row)
    tables = ((PREFIX + "_registration.csv", candidate_rows),
              ("registration_with_candidates.csv", all_rows),
              ("rgb_candidate_manifest.csv", [row for row in candidate_rows if row["status"] == "candidate_rgb"]))
    for name, rows in tables:
        path = Path(output) / name
        temporary = path.with_name(path.name + ".candidate.tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--output", type=Path, default=Path("E:/Dataset/SEO/nir_aligned_v1"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keys", help="Optional comma-separated subset of strictly rejected keys for a pilot")
    parser.add_argument("--no-loftr", action="store_true")
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("Candidate recovery uses one fresh process per image; set --workers 1")
    validate_output(args.root, args.output)
    original_path = args.root / "nir_aligned_v1" / "registration.jsonl"
    strict_path = args.root / "nir_aligned_v1" / "pan_recovery_registration.jsonl"
    originals, strict = load_records(original_path), load_records(strict_path)
    pairs = select_remaining_pairs(args.root, originals, strict, args.keys.split(",") if args.keys else None)
    selected = {pair["key"] for pair in pairs}
    options = CandidateOptions(use_loftr=not args.no_loftr)
    sources = ("recover_seo_nir_candidates.py", "nir_registration_relaxed.py", "recover_seo_nir_pan.py",
               "recover_seo_nir_pan_isolated.py", "align_seo_nir.py", "nir_registration_metrics.py",
               "nir_registration_pan.py", "compare_seo_nir_registration.py",
               "nir_registration_benchmark/pan_fullres_probe.py")
    config = {"version": VERSION, "source_root": str(args.root.resolve()), "output": str(args.output.resolve()),
              "selected_keys": sorted(selected), "options": asdict(options),
              "original_manifest_sha256": sha256(original_path), "strict_manifest_sha256": sha256(strict_path),
              "source_sha256": {name: sha256(HERE / name) for name in sources},
              "execution_mode": "fresh_single_worker_per_image", "quality_status": QUALITY}
    if options.use_loftr:
        config["weights_sha256"] = sha256(options.weights)
    # JSON has lists rather than tuples; compare the same canonical form on resume.
    config = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / (PREFIX + "_config.json")
    baseline_path = args.output / (PREFIX + "_baseline.json")
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        baseline_hash = existing.pop("baseline_sha256")
        if existing != config or not baseline_path.is_file() or sha256(baseline_path) != baseline_hash:
            raise ValueError("Candidate configuration/source/baseline changed; do not mix runs")
        config["baseline_sha256"] = baseline_hash
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        for pair in pairs:
            nir_path, valid_path = product_paths(pair, args.output)
            if nir_path.exists() or valid_path.exists():
                raise FileExistsError("A selected candidate already has an unrecognized output: " + pair["key"])
        baseline = capture_baseline(args.output, selected)
        atomic_json(baseline_path, baseline)
        config["baseline_sha256"] = sha256(baseline_path)
        atomic_json(config_path, config)
    issues = check_baseline(args.output, baseline)
    if issues:
        raise ValueError("Historical preservation check failed: " + "; ".join(issues[:5]))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = args.output / (PREFIX + "_registration.jsonl")
    records = load_records(manifest)
    jobs = [pair for pair in pairs if not reusable(records.get(pair["key"]), pair, args.output, config_hash)]
    started = time.monotonic()
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.flush()
        with IsolatedImageExecutor(max_workers=1) as executor:
            futures = {executor.submit(process_pair, pair, args.output, options, None, True, config_hash): pair for pair in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"key": pair["key"], "status": "error", "version": VERSION,
                              "config_hash": config_hash, "quality_status": QUALITY,
                              "registration_verified": False, "current_rgb_ready": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
                records[pair["key"]] = record
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                if index % 10 == 0 or index == len(jobs):
                    progress = summarize(records, selected)
                    progress.update(processed_this_run=index, jobs_this_run=len(jobs),
                                    elapsed_seconds=round(time.monotonic() - started, 1))
                    print(json.dumps(progress), flush=True)
                    write_tables(args.output, originals, strict, records, selected)
    summary = summarize(records, selected)
    summary.update(attempted_this_run=len(jobs), output=str(args.output),
                   elapsed_seconds=round(time.monotonic() - started, 1))
    issues = check_baseline(args.output, baseline)
    summary["historical_integrity_ok"] = not issues
    summary["historical_integrity_issues"] = issues
    write_tables(args.output, originals, strict, records, selected)
    atomic_json(args.output / (PREFIX + "_summary.json"), summary)
    print(json.dumps(summary), flush=True)
    return 1 if summary["errors"] or summary["retry_keys"] or issues else 0


def summarize(records, selected):
    current = [records[key] for key in selected if key in records]
    return {"selected": len(selected), "done": len(current),
            "saved": sum(record["status"] in SAVED for record in current),
            "candidate_rgb": sum(record["status"] == "candidate_rgb" for record in current),
            "candidate_pan": sum(record["status"] == "candidate_pan" for record in current),
            "rejected": sum(record["status"] == "rejected" for record in current),
            "errors": sum(record["status"] == "error" for record in current),
            "retry_keys": sorted(record["key"] for record in current if record["status"] == "error"
                                 or execution_events(record) or record.get("preview_error"))}


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

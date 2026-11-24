"""Read-only final audit of mixed original and PAN-recovered S-EO NIR products.

Run after recovery has finished. The default audit enumerates only AOI output
folders, checks every new TIFF/PNG header, and decodes a bounded stratified
sample. It never changes dataset files or writes a report into the dataset.
Exit codes: 0 consistent; 1 consistent but retryable execution errors remain;
2 integrity checks failed. Printed counts include all records, examples are capped.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import MaskFlags
from rasterio.errors import NotGeoreferencedWarning


HISTORICAL_CONFIG_SHA256 = "b503b4094da49c0af70d8a7f5d020d9dca2d316ff9b8d91616ceac9d03a98b38"
SAVED = {"rgb_ready", "pan_only"}
TERMINAL = SAVED | {"rejected", "error"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_jsonl(path):
    records, malformed, blanks = {}, [], 0
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                blanks += 1
                continue
            try:
                record = json.loads(line)
                records[record["key"]] = record
            except (ValueError, KeyError, TypeError):
                malformed.append(number)
    return records, {"blank_lines": blanks, "malformed_nonblank_lines": malformed}


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError(f"Expected explicit true/false, found {value!r}")
    return str(value).lower() == "true"


def same_path(first, second):
    return os.path.normcase(os.path.normpath(str(first))) == os.path.normcase(os.path.normpath(str(second)))


def paths_for(output, key):
    relative = Path(key)
    if len(relative.parts) != 2 or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe or non-AOI key: {key}")
    return output / (key + "_nir.tif"), output / (key + "_valid.png")


def execution_events(value, prefix=""):
    events = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + "." + key if prefix else key
            if key == "execution_error" and child:
                events.append({"path": path, "error": child, "reason": value.get("reason")})
            else:
                events.extend(execution_events(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            events.extend(execution_events(child, f"{prefix}[{index}]"))
    return events


def audit(root, output, expected_ok=19570, expected_total=20275, expected_selected=705,
          expected_v1_config_sha256=HISTORICAL_CONFIG_SHA256, samples_per_grid=3, decode_all=False):
    """Return a JSON-safe report. All file access is read-only."""
    root, output = Path(root), Path(output)
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    original_path = output / "registration.jsonl"
    config_path = output / "processing_config.json"
    recovery_config = json.loads((output / "pan_recovery_config.json").read_text(encoding="utf-8"))
    check(same_path(recovery_config.get("source_root", ""), root), "Recovery configuration points to another source root")
    check(same_path(recovery_config.get("output", ""), output), "Recovery configuration points to another output directory")
    config_hash = hashlib.sha256(json.dumps(recovery_config, sort_keys=True).encode()).hexdigest()
    original_sha = sha256(original_path)
    historical_sha = sha256(config_path)
    check(original_sha == recovery_config["original_manifest_sha256"], "Historical v1 registration.jsonl hash changed")
    check(historical_sha == expected_v1_config_sha256.lower(), "Historical v1 processing_config.json hash changed")
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    check(old_config.get("version") == "seo-nir-rgb-grid-v1", "Historical v1 configuration version changed")
    original, old_lines = latest_jsonl(original_path)
    recovered, new_lines = latest_jsonl(output / "pan_recovery_registration.jsonl")
    check(not old_lines["malformed_nonblank_lines"], "Malformed nonblank historical manifest lines")
    check(not new_lines["malformed_nonblank_lines"], "Malformed nonblank recovery manifest lines")
    original_keys = set(original)
    old_ok = {key for key, record in original.items() if record.get("status") == "ok"}
    selected = set(recovery_config["selected_keys"])
    check(len(original) == expected_total, f"Historical key count is {len(original)}, expected {expected_total}")
    check(len(old_ok) == expected_ok, f"Historical ok count is {len(old_ok)}, expected {expected_ok}")
    check(len(selected) == expected_selected, f"Selected recovery key count is {len(selected)}, expected {expected_selected}")
    check(selected == original_keys - old_ok, "Selected set differs from all historical non-ok keys")
    check(not old_ok.intersection(recovered), "Recovery wrote a key belonging to an original ok product")
    check(set(recovered) == selected, "Recovery latest-key set differs from the fixed selected set")
    status_counts = Counter(record.get("status") for record in recovered.values())
    check(set(status_counts) <= TERMINAL, "Unexpected recovery terminal status")
    saved = {key for key, record in recovered.items() if record.get("status") in SAVED}
    rgb_ready = {key for key, record in recovered.items() if record.get("status") == "rgb_ready"}
    expected_files = old_ok | saved
    expected_training = old_ok | rgb_ready

    source_checks = {}
    for name, expected in recovery_config.get("source_sha256", {}).items():
        path = Path(__file__).resolve().parent / name
        source_checks[name] = path.exists() and sha256(path) == expected
        check(source_checks[name], f"Frozen source fingerprint changed: {name}")

    csv_report = {}
    all_rows_by_key = {}
    for name, expected_keys in (("registration_all.csv", original_keys),
                                ("rgb_training_manifest.csv", expected_training),
                                ("pan_recovery_registration.csv", selected)):
        rows = csv_rows(output / name)
        counts = Counter(row["key"] for row in rows)
        keys = set(counts)
        check(len(rows) == len(keys), f"Duplicate keys in {name}")
        check(keys == expected_keys, f"Key set mismatch in {name}")
        if name == "registration_all.csv":
            all_rows_by_key = {row["key"]: row for row in rows}
        for row in rows:
            key = row["key"]
            ready = key in expected_training
            try:
                check(truth(row.get("current_rgb_ready")) == ready, f"Wrong RGB-ready flag in {name}: {key}")
            except ValueError as exc:
                check(False, f"{name} {key}: {exc}")
            if key in recovered:
                check(row.get("status") == recovered[key]["status"], f"Wrong status in {name}: {key}")
            if key in old_ok:
                check(row.get("reference_grid") == "RGB", f"Original product not marked RGB in {name}: {key}")
                for field in ("nir", "valid"):
                    check(same_path(row.get(field, ""), original[key].get(field, "")),
                          f"Original product path changed in {name}: {key} ({field})")
            elif key in saved:
                grid = "RGB" if key in rgb_ready else "PAN"
                check(row.get("reference_grid") == grid, f"Wrong grid in {name}: {key}")
            elif key in recovered:
                check(not row.get("nir") and not row.get("valid"), f"Unsuccessful key has a CSV product path in {name}: {key}")
        csv_report[name] = {"rows": len(rows), "unique_keys": len(keys)}

    actual_nir, actual_valid = set(), set()
    with os.scandir(output) as entries:
        aoi_dirs = [Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)
                    and re.fullmatch(r"(?:JAX|OMA|UCSD)_\d+", entry.name)]
    for parent in aoi_dirs:
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.endswith("_nir.tif"):
                    actual_nir.add(parent.name + "/" + entry.name[:-8])
                elif entry.name.endswith("_valid.png"):
                    actual_valid.add(parent.name + "/" + entry.name[:-10])
    check(actual_nir == expected_files, "Actual NIR key set differs from original ok plus recovered saved")
    check(actual_valid == expected_files, "Actual valid-PNG key set differs from original ok plus recovered saved")
    rejected = {key for key, record in recovered.items() if record.get("status") == "rejected"}
    check(not rejected.intersection(actual_nir | actual_valid), "Rejected keys have main NIR/valid products")

    header_checked = 0
    for key, record in recovered.items():
        check(record.get("config_hash") == config_hash, f"Wrong recovery configuration hash: {key}")
        if key not in saved:
            continue
        product = record.get("product", {})
        grid = "RGB" if key in rgb_ready else "PAN"
        check(record.get("pan_accepted") is True, f"Saved product lacks accepted PAN QA: {key}")
        check(record.get("reference_grid") == grid, f"Wrong record grid: {key}")
        check(record.get("rgb_ready") is (grid == "RGB"), f"Wrong record RGB-ready flag: {key}")
        check(product.get("reference_grid") == grid, f"Wrong product grid: {key}")
        check(product.get("current_rgb_ready") is (grid == "RGB"), f"Wrong product RGB-ready flag: {key}")
        check(record.get("version") == recovery_config["version"], f"Wrong recovery version: {key}")
        try:
            nir_path, valid_path = paths_for(output, key)
            check(same_path(product.get("nir", ""), nir_path), f"Unexpected NIR path: {key}")
            check(same_path(product.get("valid", ""), valid_path), f"Unexpected valid-PNG path: {key}")
            row = all_rows_by_key.get(key, {})
            check(same_path(row.get("nir", ""), nir_path), f"CSV points to another NIR path: {key}")
            check(same_path(row.get("valid", ""), valid_path), f"CSV points to another valid-PNG path: {key}")
            check(nir_path.stat().st_size == product["output_bytes"], f"NIR byte size changed: {key}")
            check(valid_path.stat().st_size == product["valid_bytes"], f"Valid-PNG byte size changed: {key}")
            check(0.1 <= product.get("valid_fraction", 0) <= 1, f"Invalid recorded support fraction: {key}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(nir_path, "r") as dataset:
                    tags = dataset.tags()
                    shape = (dataset.width, dataset.height)
                    check(dataset.count == 1 and dataset.dtypes == ("float32",), f"Wrong NIR band/dtype: {key}")
                    check(dataset.nodata is not None and np.isnan(dataset.nodata), f"NIR nodata is not NaN: {key}")
                    check(shape == tuple(product["target_size_wh"]), f"Wrong NIR dimensions: {key}")
                    check(tags.get("REFERENCE_GRID") == grid, f"Wrong TIFF reference-grid tag: {key}")
                    check(tags.get("CURRENT_RGB_READY") == str(grid == "RGB").lower(), f"Wrong TIFF RGB-ready tag: {key}")
                    check(tags.get("PIPELINE_VERSION") == recovery_config["version"], f"Wrong TIFF pipeline version: {key}")
                    check(tags.get("RECOVERY_CONFIG") == config_hash, f"Wrong TIFF configuration tag: {key}")
                    check(tags.get("UNITS") == "raw_DN", f"Wrong NIR units: {key}")
                    relative = Path(key)
                    for tag, parent, suffix in (("SOURCE_MSI", "msi", "_msi.tif"),
                                                ("REFERENCE_PAN", "crops", "_pan.tif"),
                                                ("REFERENCE_RGB", "pansharpened_he", "_rgb.png")):
                        expected_path = root / parent / relative.parent / (relative.name + suffix)
                        check(same_path(tags.get(tag, ""), expected_path), f"Wrong TIFF reference path {tag}: {key}")
                    matrix = np.asarray(json.loads(tags.get("H_NATIVE_TO_TARGET", "null")), np.float64)
                    check(matrix.shape == (3, 3) and np.isfinite(matrix).all(), f"Malformed saved homography: {key}")
                    if matrix.shape == (3, 3):
                        check(np.allclose(matrix, record.get("H_native_to_target"), rtol=0, atol=1e-9),
                              f"TIFF and record homographies differ: {key}")
                    check(MaskFlags.per_dataset in dataset.mask_flag_enums[0], f"Missing dataset validity mask: {key}")
                    check(not any(path.endswith(".msk") for path in dataset.files), f"Unexpected external GDAL mask: {key}")
            with Image.open(valid_path) as image:
                check(image.size == shape and image.mode == "L", f"Wrong validity-PNG header: {key}")
            header_checked += 1
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            check(False, f"Cannot inspect product {key}: {type(exc).__name__}: {exc}")

    decode_keys = set(saved) if decode_all else set()
    if not decode_all:
        for grid in ("PAN", "RGB"):
            candidates = sorted((key for key in saved if recovered[key].get("reference_grid") == grid),
                                 key=lambda key: (recovered[key]["product"].get("valid_fraction", 0), key))
            if candidates and samples_per_grid:
                positions = np.linspace(0, len(candidates) - 1, min(samples_per_grid, len(candidates))).astype(int)
                decode_keys.update(candidates[index] for index in positions)
    decoded = []
    for key in sorted(decode_keys):
        try:
            nir_path, valid_path = paths_for(output, key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(nir_path, "r") as dataset:
                    values, valid = dataset.read(1), dataset.read_masks(1) > 0
            with Image.open(valid_path) as image:
                png = np.asarray(image)
            check(set(np.unique(png)).issubset({0, 255}), f"Nonbinary PNG validity values: {key}")
            check(np.array_equal(valid, png > 0), f"Internal/PNG validity masks differ: {key}")
            check(np.isfinite(values[valid]).all(), f"Nonfinite DN inside valid support: {key}")
            check(np.isnan(values[~valid]).all(), f"Invalid output values are not NaN: {key}")
            product = recovered[key]["product"]
            check(abs(float(valid.mean()) - product["valid_fraction"]) < 1e-9, f"Actual validity fraction differs: {key}")
            if valid.any():
                check(np.isclose(float(values[valid].min()), product["nir_min"], rtol=1e-6, atol=1e-3), f"NIR minimum differs: {key}")
                check(np.isclose(float(values[valid].max()), product["nir_max"], rtol=1e-6, atol=1e-3), f"NIR maximum differs: {key}")
            decoded.append({"key": key, "reference_grid": recovered[key]["reference_grid"], "valid_fraction": float(valid.mean())})
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            check(False, f"Cannot decode product {key}: {type(exc).__name__}: {exc}")

    summary = json.loads((output / "pan_recovery_summary.json").read_text(encoding="utf-8"))
    expected_summary = {"selected": len(selected), "saved": len(saved), "rgb_ready": len(rgb_ready),
                        "pan_only": status_counts["pan_only"], "rejected": status_counts["rejected"], "errors": status_counts["error"]}
    for name, expected in expected_summary.items():
        check(summary.get(name) == expected, f"Recovery summary count mismatch: {name}")
    errors = [{"key": key, "reason": record.get("reason")} for key, record in recovered.items() if record.get("status") == "error"]
    preview_errors = [{"key": key, "error": record["preview_error"]} for key, record in recovered.items() if record.get("preview_error")]
    attempts = []
    bridges = []
    for key, record in recovered.items():
        attempts.extend({"key": key, **event} for event in execution_events(record.get("pan_registration", {}), "pan_registration"))
        bridges.extend({"key": key, **event} for event in execution_events(record.get("rgb_bridge", {}), "rgb_bridge"))
    retry_keys = {entry["key"] for entry in errors} | {
        key for key, record in recovered.items() if record.get("rgb_bridge", {}).get("execution_error")}
    return {"integrity_ok": not issues, "issues": issues, "requires_retry": bool(retry_keys), "retry_keys": sorted(retry_keys),
            "historical_manifest_sha256": original_sha, "historical_config_sha256": historical_sha,
            "historical_ok": len(old_ok), "original_total_keys": len(original), "selected": len(selected),
            "recovery_status_counts": dict(status_counts), "saved": len(saved), "new_rgb_ready": len(rgb_ready),
            "old_ok_recovery_overlap": sorted(old_ok.intersection(recovered)),
            "csv": csv_report, "actual_nir_files": len(actual_nir), "actual_valid_png_files": len(actual_valid),
            "missing_nir_keys": sorted(expected_files - actual_nir), "extra_nir_keys": sorted(actual_nir - expected_files),
            "missing_valid_keys": sorted(expected_files - actual_valid), "extra_valid_keys": sorted(actual_valid - expected_files),
            "original_jsonl": old_lines, "recovery_jsonl": new_lines, "frozen_source_checks": source_checks,
            "new_headers_checked": header_checked, "decoded_samples": decoded,
            "execution_errors": errors, "preview_errors": preview_errors,
            "pan_attempt_execution_errors": attempts, "bridge_execution_errors": bridges,
            "scope": "All ledger/file sets and new headers; only listed samples decoded unless --decode-all."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-v1-config-sha256", default=HISTORICAL_CONFIG_SHA256)
    parser.add_argument("--samples-per-grid", type=int, default=3)
    parser.add_argument("--decode-all", action="store_true")
    args = parser.parse_args()
    if args.samples_per_grid < 0:
        parser.error("samples-per-grid must be nonnegative")
    try:
        with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
            report = audit(args.root, args.output or args.root / "nir_aligned_v1",
                           expected_v1_config_sha256=args.expected_v1_config_sha256,
                           samples_per_grid=args.samples_per_grid, decode_all=args.decode_all)
        print(json.dumps(report, ensure_ascii=True, allow_nan=False))
        return 2 if not report["integrity_ok"] else int(report["requires_retry"])
    except Exception as exc:
        print(json.dumps({"integrity_ok": False, "fatal_error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

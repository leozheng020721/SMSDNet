"""Read-only audit of completed, explicitly unverified S-EO NIR candidates.

Exit codes: 0 consistent; 1 consistent but execution errors need retrying;
2 integrity failure. Only candidate rasters are decoded, never old products.
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

from align_seo_nir import input_fingerprint, validate_homography


HERE = Path(__file__).resolve().parent
VERSION = "seo-nir-pan-candidate-v3"
QUALITY = "UNVERIFIED_CANDIDATE"
SAVED = {"candidate_rgb", "candidate_pan"}
TERMINAL = SAVED | {"rejected", "error"}
PROTECTED_LEDGERS = {
    "registration.jsonl", "registration.csv", "processing_config.json", "summary.json",
    "pan_recovery_config.json", "pan_recovery_registration.jsonl",
    "pan_recovery_registration.csv", "pan_recovery_summary.json",
    "registration_all.csv", "rgb_training_manifest.csv",
}
FROZEN_SOURCES = {
    "recover_seo_nir_candidates.py", "nir_registration_relaxed.py", "recover_seo_nir_pan.py",
    "recover_seo_nir_pan_isolated.py", "align_seo_nir.py", "nir_registration_metrics.py",
    "nir_registration_pan.py", "compare_seo_nir_registration.py",
    "nir_registration_benchmark/pan_fullres_probe.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_path(first, second):
    return os.path.normcase(os.path.abspath(str(first))) == os.path.normcase(os.path.abspath(str(second)))


def latest_jsonl(path):
    records, malformed = {}, []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("key"), str):
                    raise ValueError("Expected keyed JSON object")
                records[row["key"]] = row
            except (ValueError, TypeError):
                malformed.append(number)
    return records, malformed


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value):
    lowered = str(value).lower()
    if lowered not in ("true", "false"):
        raise ValueError("Expected explicit boolean, received " + repr(value))
    return lowered == "true"


def paths_for(output, key):
    relative = Path(key)
    if (len(relative.parts) != 2 or relative.is_absolute() or ".." in relative.parts
            or not re.fullmatch(r"(?:JAX|OMA|UCSD)_\d+", relative.parts[0])):
        raise ValueError("Unsafe AOI key: " + key)
    return Path(output) / (key + "_nir.tif"), Path(output) / (key + "_valid.png")


def source_pair(root, key):
    relative = Path(key)
    return {name: str(Path(root) / folder / relative.parent / (relative.name + suffix))
            for name, folder, suffix in (("msi", "msi", "_msi.tif"),
                                         ("pan", "crops", "_pan.tif"),
                                         ("rgb", "pansharpened_he", "_rgb.png"))}


def execution_events(value, prefix=""):
    events = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + "." + key if prefix else key
            if key == "execution_error" and child:
                events.append({"path": path, "error": str(child)})
            else:
                events.extend(execution_events(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            events.extend(execution_events(child, f"{prefix}[{index}]"))
    return events


def normalized_h(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-10:
        raise ValueError("Malformed homography")
    return matrix / matrix[2, 2]


def audit(root, output, expected_selected=688, expected_protected_nir=19587,
          expected_total=20275, expected_original_ok=19570, expected_strict_saved=17,
          expected_training=19584, allow_pilot=False, samples_per_grid=3, decode_all=False):
    """Inspect ledgers, all candidate headers, and selected candidate pixel data."""
    root, output = Path(root).resolve(), Path(output).resolve()
    historical = root / "nir_aligned_v1"
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    config = json.loads((output / "pan_candidate_config.json").read_text(encoding="utf-8"))
    baseline_path = output / "pan_candidate_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    check(config.get("version") == VERSION, "Unexpected candidate pipeline version")
    check(config.get("quality_status") == QUALITY, "Configuration lacks unverified quality label")
    check(same_path(config.get("source_root", ""), root), "Configuration source root differs")
    check(same_path(config.get("output", ""), output), "Configuration output differs")
    check(sha256(baseline_path) == config.get("baseline_sha256"), "Protected baseline SHA256 changed")
    if not allow_pilot:
        check(same_path(output, historical), "Production audit must use the original output directory")

    original_path = historical / "registration.jsonl"
    strict_path = historical / "pan_recovery_registration.jsonl"
    check(sha256(original_path) == config.get("original_manifest_sha256"), "Original manifest SHA256 changed")
    check(sha256(strict_path) == config.get("strict_manifest_sha256"), "Strict manifest SHA256 changed")
    original, original_malformed = latest_jsonl(original_path)
    strict, strict_malformed = latest_jsonl(strict_path)
    records, candidate_malformed = latest_jsonl(output / "pan_candidate_registration.jsonl")
    check(not original_malformed, "Malformed historical original JSONL lines")
    check(not strict_malformed, "Malformed historical strict JSONL lines")
    original_keys = set(original)
    old_ok = {key for key, row in original.items() if row.get("status") == "ok"}
    strict_saved = {key for key, row in strict.items() if row.get("status") in {"rgb_ready", "pan_only"}}
    strict_rgb = {key for key, row in strict.items() if row.get("status") == "rgb_ready"}
    strict_rejected = {key for key, row in strict.items() if row.get("status") == "rejected"
                       and key in original_keys - old_ok}
    protected_keys = old_ok | strict_saved
    selected_list = config.get("selected_keys", [])
    selected = set(selected_list)
    check(len(selected_list) == len(selected), "Duplicate configured selected keys")
    check(len(original_keys) == expected_total, "Unexpected original total key count")
    check(len(old_ok) == expected_original_ok, "Unexpected original accepted key count")
    check(len(strict_saved) == expected_strict_saved, "Unexpected strict saved key count")
    check(set(strict) == original_keys - old_ok, "Strict key set differs from original failed cohort")
    check(not old_ok & strict_saved, "Strict recovery overlaps original accepted keys")
    check(len(selected) == expected_selected, "Unexpected configured candidate key count")
    check(selected <= strict_rejected if allow_pilot else selected == strict_rejected,
          "Candidate selection differs from the strictly rejected cohort")
    check(set(records) == selected, "Latest candidate JSONL keys differ from configured cohort")
    check(not set(records) & protected_keys, "Candidate JSONL overlaps previously accepted products")
    counts = Counter(row.get("status") for row in records.values())
    check(set(counts) <= TERMINAL, "Unexpected candidate terminal status")
    saved = {key for key, row in records.items() if row.get("status") in SAVED}
    candidate_rgb = {key for key, row in records.items() if row.get("status") == "candidate_rgb"}

    source_checks = {}
    sources = config.get("source_sha256", {})
    check(set(sources) == FROZEN_SOURCES, "Frozen source manifest is incomplete or unexpected")
    for name, digest in sources.items():
        source = HERE / name
        source_checks[name] = source.is_file() and sha256(source) == digest
        check(source_checks[name], "Frozen source SHA256 changed: " + name)
    if config.get("options", {}).get("use_loftr"):
        weights = Path(config["options"]["weights"])
        check(weights.is_file() and sha256(weights) == config.get("weights_sha256"), "LoFTR weights SHA256 changed")

    protected = baseline.get("protected_products", {})
    ledgers = baseline.get("protected_ledgers_sha256", {})
    if not allow_pilot:
        check(set(ledgers) == PROTECTED_LEDGERS, "Historical ledger baseline is incomplete")
    for name, digest in ledgers.items():
        path = output / name
        check(name in PROTECTED_LEDGERS, "Unexpected protected ledger: " + name)
        check(path.is_file() and sha256(path) == digest, "Historical ledger SHA256 changed: " + name)
    baseline_nir, baseline_valid = set(), set()
    for name, metadata in protected.items():
        path = output / name
        check(path.is_file(), "Historical product missing: " + name)
        if path.is_file():
            info = path.stat()
            check(info.st_size == metadata.get("size") and info.st_mtime_ns == metadata.get("mtime_ns"),
                  "Historical product size/mtime changed: " + name)
        if name.endswith("_nir.tif"):
            baseline_nir.add(name[:-8])
        elif name.endswith("_valid.png"):
            baseline_valid.add(name[:-10])
        else:
            check(False, "Unexpected protected product suffix: " + name)
    check(baseline_nir == baseline_valid, "Protected NIR and validity-mask baseline key sets differ")
    check(len(baseline_nir) == expected_protected_nir, "Unexpected protected NIR baseline count")
    check(baseline.get("protected_nir_count") == len(baseline_nir), "Recorded protected NIR count differs")
    check(baseline.get("protected_valid_count") == len(baseline_valid), "Recorded protected mask count differs")
    check(not baseline_nir & selected, "Candidate cohort overlaps protected product baseline")
    if not allow_pilot:
        check(baseline_nir == protected_keys, "Protected product baseline differs from previous accepted cohort")

    csv_report, unified = {}, {}
    table_specs = ((output / "registration_with_candidates.csv", original_keys),
                   (output / "pan_candidate_registration.csv", selected),
                   (output / "rgb_candidate_manifest.csv", candidate_rgb),
                   (historical / "rgb_training_manifest.csv", old_ok | strict_rgb))
    for path, expected_keys in table_specs:
        rows = csv_rows(path)
        keys = {row["key"] for row in rows}
        check(len(rows) == len(keys), "Duplicate keys in " + path.name)
        check(keys == expected_keys, "Key set differs in " + path.name)
        if path.name == "rgb_training_manifest.csv":
            check(len(rows) == expected_training, "Historical RGB training count differs")
            check(not keys & selected, "Candidate leaked into historical RGB training manifest")
        if path.name == "registration_with_candidates.csv":
            unified = {row["key"]: row for row in rows}
        for row in rows:
            key = row["key"]
            try:
                if key in records:
                    record = records[key]
                    check(row.get("status") == record.get("status"), "CSV candidate status differs: " + key)
                    check(row.get("quality_status") == QUALITY, "CSV candidate quality label differs: " + key)
                    check(not truth(row.get("registration_verified")), "CSV candidate marked verified: " + key)
                    check(not truth(row.get("current_rgb_ready")), "CSV candidate marked RGB ready: " + key)
                    if key in saved:
                        product = record["product"]
                        check(row.get("reference_grid") == product["reference_grid"], "CSV candidate grid differs: " + key)
                        check(truth(row.get("rgb_grid_verified")) == product.get("rgb_grid_verified"),
                              "CSV RGB bridge verification flag differs: " + key)
                        for field in ("nir", "valid"):
                            check(same_path(row.get(field, ""), product[field]), "CSV candidate path differs: " + key)
                    else:
                        check(not row.get("nir") and not row.get("valid"), "Unsuccessful CSV row has product paths: " + key)
                elif path.name == "registration_with_candidates.csv":
                    ready = key in old_ok | strict_rgb
                    check(truth(row.get("current_rgb_ready")) == ready, "Unified historical RGB-ready flag differs: " + key)
                    check(truth(row.get("registration_verified")) == (key in protected_keys),
                          "Unified historical accepted flag differs: " + key)
            except (KeyError, ValueError, TypeError) as exc:
                check(False, f"Cannot validate CSV {path.name} {key}: {exc}")
        csv_report[path.name] = {"rows": len(rows), "unique_keys": len(keys)}

    actual_nir, actual_valid, temporary_files = set(), set(), []
    for parent in output.iterdir():
        if parent.is_file():
            if parent.name.endswith(".tmp"):
                temporary_files.append(parent.name)
            continue
        if not parent.is_dir() or not re.fullmatch(r"(?:JAX|OMA|UCSD)_\d+", parent.name):
            continue
        for path in parent.iterdir():
            if path.is_file():
                relative = path.relative_to(output).as_posix()
                if path.name.endswith("_nir.tif"):
                    actual_nir.add(relative[:-8])
                elif path.name.endswith("_valid.png"):
                    actual_valid.add(relative[:-10])
                elif path.name.endswith(".tmp"):
                    temporary_files.append(relative)
    expected_files = baseline_nir | saved
    check(actual_nir == expected_files, "Actual NIR keys differ from protected plus saved candidates")
    check(actual_valid == expected_files, "Actual validity keys differ from protected plus saved candidates")
    check(not temporary_files, "Unfinished temporary files remain")
    check(not (selected - saved) & (actual_nir | actual_valid), "Unsuccessful candidate has a main product")

    header_checked = 0
    for key, record in records.items():
        check(record.get("version") == VERSION, "Wrong record pipeline version: " + key)
        check(record.get("config_hash") == config_hash, "Wrong record configuration hash: " + key)
        check(record.get("quality_status") == QUALITY, "Wrong record candidate label: " + key)
        check(record.get("registration_verified") is False, "Record marked registration verified: " + key)
        check(record.get("current_rgb_ready") is False, "Record marked RGB ready: " + key)
        pair = source_pair(root, key)
        try:
            if record.get("inputs") is not None:
                check(record["inputs"] == input_fingerprint(pair), "Source fingerprint changed: " + key)
            if key not in saved:
                continue
            nir_path, valid_path = paths_for(output, key)
            product = record["product"]
            grid = "RGB" if key in candidate_rgb else "PAN"
            check(record.get("reference_grid") == grid, "Wrong candidate record grid: " + key)
            check(product.get("reference_grid") == grid, "Wrong candidate product grid: " + key)
            check(product.get("quality_status") == QUALITY, "Wrong product quality label: " + key)
            check(product.get("registration_verified") is False, "Product marked registration verified: " + key)
            check(product.get("current_rgb_ready") is False, "Product marked RGB ready: " + key)
            check(product.get("rgb_grid_verified") is (grid == "RGB"), "Product bridge flag differs: " + key)
            check(record.get("pan_registration", {}).get("accepted") is True, "Saved candidate lacks accepted PAN fit: " + key)
            if grid == "RGB":
                check(record.get("rgb_bridge", {}).get("accepted") is True, "RGB candidate lacks accepted bridge: " + key)
            for field, expected_path in (("nir", nir_path), ("valid", valid_path)):
                check(same_path(product.get(field, ""), expected_path), "Unexpected candidate product path: " + key)
            check(nir_path.stat().st_size == product["output_bytes"], "Candidate NIR byte size changed: " + key)
            check(valid_path.stat().st_size == product["valid_bytes"], "Candidate mask byte size changed: " + key)
            check(0.1 <= product.get("valid_fraction", 0) <= 1, "Candidate valid coverage outside limits: " + key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(pair["msi"]) as source:
                    source_size = (source.width, source.height)
                    check(source.count == 4, "Source is not four-band MSI: " + key)
                if grid == "PAN":
                    with rasterio.open(pair["pan"]) as target:
                        target_size = (target.width, target.height)
                else:
                    with Image.open(pair["rgb"]) as target:
                        target_size = target.size
                matrix = validate_homography(record["H_native_to_target"], source_size, target_size)
                pan_matrix = normalized_h(record["H_native_to_pan"])
                expected_h = normalized_h(record["rgb_bridge"]["H_pan_to_rgb"]) @ pan_matrix if grid == "RGB" else pan_matrix
                check(np.allclose(normalized_h(matrix), normalized_h(expected_h), rtol=1e-9, atol=1e-8),
                      "Saved matrix differs from PAN/bridge composition: " + key)
                with rasterio.open(nir_path) as dataset:
                    tags = dataset.tags()
                    check(dataset.count == 1 and dataset.dtypes == ("float32",), "Wrong candidate dtype/band count: " + key)
                    check(dataset.nodata is not None and np.isnan(dataset.nodata), "Candidate nodata is not NaN: " + key)
                    check((dataset.width, dataset.height) == target_size == tuple(product["target_size_wh"]),
                          "Candidate target dimensions differ: " + key)
                    expected_tags = {"PIPELINE_VERSION": VERSION, "RECOVERY_CONFIG": config_hash,
                                     "REGISTRATION_STATUS": QUALITY, "REGISTRATION_VERIFIED": "false",
                                     "CURRENT_RGB_READY": "false", "REFERENCE_GRID": grid,
                                     "RGB_GRID_VERIFIED": str(grid == "RGB").lower(), "UNITS": "raw_DN"}
                    for name, expected in expected_tags.items():
                        check(tags.get(name) == expected, f"Wrong candidate TIFF {name}: {key}")
                    for tag, name in (("SOURCE_MSI", "msi"), ("REFERENCE_PAN", "pan"), ("REFERENCE_RGB", "rgb")):
                        check(same_path(tags.get(tag, ""), pair[name]), f"Wrong TIFF source {tag}: {key}")
                    tagged_h = normalized_h(json.loads(tags["H_NATIVE_TO_TARGET"]))
                    check(np.allclose(tagged_h, normalized_h(matrix), rtol=1e-9, atol=1e-8), "TIFF matrix differs from record: " + key)
                    check(MaskFlags.per_dataset in dataset.mask_flag_enums[0], "Missing candidate internal mask: " + key)
                    check(not any(name.endswith(".msk") for name in dataset.files), "Candidate has external mask: " + key)
                with Image.open(valid_path) as image:
                    check(image.mode == "L" and image.size == target_size, "Wrong candidate PNG header: " + key)
            header_checked += 1
        except (OSError, ValueError, KeyError, TypeError, rasterio.errors.RasterioError) as exc:
            check(False, f"Cannot inspect candidate {key}: {type(exc).__name__}: {exc}")

    decode_keys = set(saved) if decode_all else set()
    if not decode_all and samples_per_grid:
        for status in SAVED:
            keys = sorted((key for key in saved if records[key]["status"] == status),
                          key=lambda key: (records[key]["product"].get("valid_fraction", 0), key))
            positions = np.linspace(0, len(keys) - 1, min(samples_per_grid, len(keys))).astype(int) if keys else []
            decode_keys.update(keys[index] for index in positions)
    decoded = []
    for key in sorted(decode_keys):
        try:
            nir_path, valid_path = paths_for(output, key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(nir_path) as dataset:
                    values, valid = dataset.read(1), dataset.read_masks(1) > 0
                with rasterio.open(source_pair(root, key)["msi"]) as source:
                    raw, source_valid = source.read(4), source.read_masks(4) > 0
            with Image.open(valid_path) as image:
                png = np.asarray(image)
            check(set(np.unique(png)) <= {0, 255}, "Nonbinary candidate PNG values: " + key)
            check(np.array_equal(valid, png > 0), "Candidate internal/PNG masks differ: " + key)
            check(np.isfinite(values[valid]).all(), "Nonfinite valid candidate DN: " + key)
            check(np.isnan(values[~valid]).all(), "Invalid candidate DN is not NaN: " + key)
            product = records[key]["product"]
            fraction = float(valid.mean())
            check(fraction >= 0.1 and abs(fraction - product["valid_fraction"]) < 1e-9,
                  "Decoded candidate valid fraction differs: " + key)
            if valid.any():
                low, high = float(values[valid].min()), float(values[valid].max())
                check(np.isclose(low, product["nir_min"], rtol=1e-6, atol=1e-3), "Candidate minimum differs: " + key)
                check(np.isclose(high, product["nir_max"], rtol=1e-6, atol=1e-3), "Candidate maximum differs: " + key)
                raw_values = raw[source_valid & np.isfinite(raw)]
                check(bool(len(raw_values)) and low >= float(raw_values.min()) - 1e-3
                      and high <= float(raw_values.max()) + 1e-3, "Candidate DN is outside source DN range: " + key)
            decoded.append({"key": key, "reference_grid": records[key]["reference_grid"], "valid_fraction": fraction})
        except (OSError, ValueError, KeyError, TypeError, rasterio.errors.RasterioError) as exc:
            check(False, f"Cannot decode candidate {key}: {type(exc).__name__}: {exc}")

    errors, preview_errors, events = [], [], []
    retry_keys = set()
    for key, record in records.items():
        if record.get("status") == "error":
            errors.append({"key": key, "reason": record.get("reason")})
            retry_keys.add(key)
        if record.get("preview_error"):
            preview_errors.append({"key": key, "error": record["preview_error"]})
            retry_keys.add(key)
        found = execution_events(record)
        if found:
            retry_keys.add(key)
            events.extend({"key": key, **event} for event in found)
    summary = json.loads((output / "pan_candidate_summary.json").read_text(encoding="utf-8"))
    expected_summary = {"selected": len(selected), "done": len(records), "saved": len(saved),
                        "candidate_rgb": len(candidate_rgb), "candidate_pan": counts["candidate_pan"],
                        "rejected": counts["rejected"], "errors": counts["error"]}
    for name, expected in expected_summary.items():
        check(summary.get(name) == expected, "Candidate summary count differs: " + name)
    check(summary.get("historical_integrity_ok") is True, "Run did not report intact historical baseline")
    check(set(summary.get("retry_keys", [])) == retry_keys, "Summary retry keys differ from execution events")
    return {"integrity_ok": not issues, "issues": issues, "requires_retry": bool(retry_keys),
            "retry_keys": sorted(retry_keys), "candidate_config_hash": config_hash,
            "selected": len(selected), "done": len(records), "saved": len(saved),
            "candidate_rgb": len(candidate_rgb), "candidate_pan": counts["candidate_pan"],
            "rejected": counts["rejected"], "errors": counts["error"],
            "historical_original_ok": len(old_ok), "historical_strict_saved": len(strict_saved),
            "protected_nir_count": len(baseline_nir), "protected_valid_count": len(baseline_valid),
            "original_total_keys": len(original_keys), "csv": csv_report,
            "actual_nir_files": len(actual_nir), "actual_valid_png_files": len(actual_valid),
            "missing_nir_keys": sorted(expected_files - actual_nir), "extra_nir_keys": sorted(actual_nir - expected_files),
            "missing_valid_keys": sorted(expected_files - actual_valid), "extra_valid_keys": sorted(actual_valid - expected_files),
            "temporary_files": temporary_files, "new_headers_checked": header_checked,
            "decoded_candidates": decoded, "frozen_source_checks": source_checks,
            "malformed_jsonl_lines": {"original": original_malformed, "strict": strict_malformed,
                                      "candidate": candidate_malformed},
            "execution_errors": errors, "attempt_execution_events": events, "preview_errors": preview_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-selected", type=int, default=688)
    parser.add_argument("--expected-protected-nir", type=int, default=19587)
    parser.add_argument("--expected-total", type=int, default=20275)
    parser.add_argument("--expected-original-ok", type=int, default=19570)
    parser.add_argument("--expected-strict-saved", type=int, default=17)
    parser.add_argument("--expected-training", type=int, default=19584)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--samples-per-grid", type=int, default=3)
    parser.add_argument("--decode-all", action="store_true")
    args = parser.parse_args()
    options = vars(args).copy()
    options["output"] = args.output or args.root / "nir_aligned_v1"
    try:
        report = audit(**options)
    except Exception as exc:
        report = {"integrity_ok": False, "issues": [f"Audit failed: {type(exc).__name__}: {exc}"],
                  "requires_retry": False, "retry_keys": []}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if not report["integrity_ok"] else 1 if report["requires_retry"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

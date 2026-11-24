"""Run one comparison sample per process to bound CPU and GDAL memory."""

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
BASE = HERE / "nir_registration_benchmark"
OUTPUT = BASE / "formal_v2"
SAMPLES_FILE = BASE / "comparison_v1" / "samples.json"
CACHE = BASE / "comparison_v1" / "pan_candidates.jsonl"
METHODS = ("direct_loftr", "pan_classical", "pan_loftr", "pan_any")


def read_result(path):
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def complete(record):
    return (record is not None and not record.get("error") and
            all(name in record.get("methods", {}) and not record["methods"][name].get("runtime_error")
                for name in METHODS))


def summary(records):
    report = {"sample_count": len(records), "sample_runtime_errors": 0,
              "by_cohort": {}, "by_city": {}}
    for group_field in ("cohort", "city"):
        groups = defaultdict(list)
        for record in records:
            groups[record[group_field]].append(record)
        for group, items in groups.items():
            report["by_" + group_field][group] = {"n": len(items)}
            for method in METHODS:
                report["by_" + group_field][group][method] = {
                    "accepted": sum(bool(r.get("methods", {}).get(method, {}).get("accepted")) for r in items),
                    "runtime_errors": sum(bool(r.get("methods", {}).get(method, {}).get("runtime_error")) for r in items)}
    report["sample_runtime_errors"] = sum(not complete(r) for r in records)
    report["pan_grid_verified"] = sum(bool(r.get("grid", {}).get("verified")) for r in records)
    report["pan_same_grid"] = sum(bool(r.get("grid", {}).get("verified") and r["grid"].get("same_grid")) for r in records)
    return report


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))
    records = []
    for index, sample in enumerate(samples, 1):
        started = time.monotonic()
        case_dir = OUTPUT / "cases" / sample["key"].replace("/", "__")
        case_dir.mkdir(parents=True, exist_ok=True)
        record = read_result(case_dir / "results.jsonl")
        for attempt in range(2):
            if complete(record):
                break
            command = [sys.executable, str(HERE / "compare_seo_nir_registration.py"),
                       "--sample-key", sample["key"], "--samples-file", str(SAMPLES_FILE),
                       "--pan-cache", str(CACHE), "--output", str(case_dir)]
            with (case_dir / ("worker_%d.log" % attempt)).open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record = read_result(case_dir / "results.jsonl")
            if process.returncode != 0 and record is None:
                record = {**sample, "error": "Worker exited %d" % process.returncode, "methods": {}}
        if record is None:
            record = {**sample, "error": "Worker produced no result", "methods": {}}
        records.append(record)
        with (OUTPUT / "results.jsonl").open("w", encoding="utf-8") as handle:
            for saved in records:
                handle.write(json.dumps(saved) + "\n")
        (OUTPUT / "summary.json").write_text(json.dumps(summary(records), indent=2), encoding="utf-8")
        print(json.dumps({"done": index, "total": len(samples), "key": sample["key"],
                          "accepted": {m: record.get("methods", {}).get(m, {}).get("accepted", False) for m in METHODS},
                          "runtime_error": not complete(record), "seconds": round(time.monotonic() - started, 2)}), flush=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ("key", "cohort", "city", "method", "accepted", "reason", "holdout_median_msi_px",
                  "holdout_p90_msi_px", "valid_fraction", "gradient_magnitude_correlation", "runtime_error")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for name in METHODS:
                result = record.get("methods", {}).get(name, {})
                row = {field: record[field] for field in ("key", "cohort", "city")}
                row.update(method=name, accepted=result.get("accepted", False), reason=result.get("reason"),
                           runtime_error=result.get("runtime_error", bool(record.get("error"))))
                row.update({k: result.get("metrics", {}).get(k) for k in ("holdout_median_msi_px", "holdout_p90_msi_px")})
                row.update({k: result.get("image_metrics", {}).get(k) for k in ("valid_fraction", "gradient_magnitude_correlation")})
                writer.writerow(row)
    print(json.dumps(summary(records)), flush=True)


if __name__ == "__main__":
    main()

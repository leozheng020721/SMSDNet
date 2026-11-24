"""Infer one registered image and evaluate an explicitly aligned binary mask."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from folder_inference import (
    TILE_FIELDS, _band, _predict_source, _tile_records, _write_png,
    discover_sources, load_model, open_source,
)


def evaluate_binary(pred, label, valid):
    """Count pixels once; BER/FNR/FPR are percentages, other scores fractions."""
    pred, label, valid = np.asarray(pred), np.asarray(label), np.asarray(valid, dtype=bool)
    if pred.ndim != 2 or pred.shape != label.shape or pred.shape != valid.shape:
        raise ValueError("Prediction, label and valid mask must have identical 2D shapes")
    pred, label = pred != 0, label != 0
    tp = int(np.count_nonzero(valid & pred & label))
    tn = int(np.count_nonzero(valid & ~pred & ~label))
    fp = int(np.count_nonzero(valid & pred & ~label))
    fn = int(np.count_nonzero(valid & ~pred & label))
    n = tp + tn + fp + fn
    if n == 0:
        raise ValueError("There are no valid labeled image pixels to evaluate")

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    fnr, fpr = divide(fn, tp + fn), divide(fp, tn + fp)
    oa = (tp + tn) / n
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "N": n,
        "shadow_prevalence": (tp + fn) / n,
        "pred_prevalence": (tp + fp) / n,
        "BER": None if fnr is None or fpr is None else (fnr + fpr) * 50.0,
        "FNR": None if fnr is None else fnr * 100.0,
        "FPR": None if fpr is None else fpr * 100.0,
        "shadowIoU": divide(tp, tp + fp + fn),
        "F1": divide(2 * tp, 2 * tp + fp + fn),
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "OA": float(oa), "Kappa": divide(oa - expected, 1.0 - expected),
    }


def read_reference(path, expected_shape, orientation="as-is"):
    """Read literal label values, including zero regardless of TIFF nodata tags."""
    if orientation not in ("as-is", "flip-ud"):
        raise ValueError("Reference orientation must be as-is or flip-ud")
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[:, :, 0]
    if raw.ndim != 2 or raw.shape != tuple(expected_shape):
        raise ValueError(f"Reference dimensions {raw.shape} differ from image {expected_shape}; no resizing is allowed")
    if orientation == "flip-ud":
        raw = np.flipud(raw).copy()
    known = np.isfinite(raw) & ((raw == 0) | (raw == 255))
    return raw == 255, known


def _signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_jpeg(path, rgb):
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise OSError(f"Could not encode JPEG: {path}")
    Path(path).write_bytes(encoded.tobytes())


def _comparison(rgb, reference, prediction, errors):
    height, width = rgb.shape[:2]
    scale = min(1.0, 850.0 / width, 1000.0 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    panels = []
    for title, array in (("RGB", rgb), ("Reference: white = shadow", reference),
                         ("Prediction: white = shadow", prediction), ("Errors: TP green, FP red, FN blue", errors)):
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        scaled = cv2.resize(array, size, interpolation=cv2.INTER_AREA if title == "RGB" else cv2.INTER_NEAREST)
        panel = np.full((size[1] + 65, size[0], 3), 255, dtype=np.uint8)
        panel[65:] = scaled
        cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        if title.startswith("Errors"):
            cv2.putText(panel, "TN black; invalid/unknown gray", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def run_evaluation(args):
    cv2.setNumThreads(2)
    tile_size = 512
    stride, batch_size = int(args.stride), int(args.tile_batch_size)
    if not 0 < stride <= tile_size or batch_size <= 0:
        raise ValueError("Require 0 < stride <= 512 and positive tile batch size")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matches = [item for item in discover_sources(args.input) if item["stem"] == args.stem]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source named {args.stem}; found {len(matches)}")
    source = matches[0]
    source["total_tiles"] = math.ceil(source["width"] / tile_size) * math.ceil(source["height"] / tile_size)
    shape = (source["height"], source["width"])
    label, known = read_reference(args.reference, shape, args.reference_orientation)
    signature = {
        "source": _signature(source["data_path"]),
        "source_header": _signature(source["source_path"]),
        "valid_mask": _signature(source["valid_data"]) if source.get("valid_data") else None,
        "valid_header": _signature(source["valid_header"]) if source.get("valid_header") else None,
        "checkpoint": _signature(args.checkpoint), "config": _signature(args.config),
        "tile_size": tile_size, "stride": stride,
        "preprocessing": "folder_inference_fixed_global_valid_rgb_percentile_2_98",
    }
    cache_path = output / "prediction_cache.json"
    prediction_path, rgb_path = output / "prediction_shadow.png", output / "rgb.png"
    reused = False
    if getattr(args, "reuse_prediction", False):
        if not cache_path.is_file() or not prediction_path.is_file() or not rgb_path.is_file():
            raise ValueError("Prediction cache is missing; first run without --reuse-prediction")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["signature"] != signature:
            raise ValueError("Prediction cache source/config/checkpoint/preprocessing signature differs")
        with Image.open(prediction_path) as image:
            pred = np.asarray(image).copy() != 0
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if pred.shape != shape or rgb.shape != shape + (3,):
            raise ValueError("Cached full-frame image dimensions differ from source")
        statistics = cached["inference"]
        resolved_config = cached["resolved_config"]
        reused = True
    else:
        print(f"Source: {source['stem']} | {shape[1]} x {shape[0]} | {source['total_tiles']} export tiles", flush=True)
        model, resolved_config, device = load_model(args.config, args.checkpoint, getattr(args, "device", None))
        tiles_root = output / "tiles"
        tiles_root.mkdir(parents=True, exist_ok=True)
        statistics = _predict_source(model, resolved_config, device, source, tiles_root,
                                     tile_size, stride, batch_size, export_layout="paired")
        del model
        pred = np.zeros(shape, dtype=bool)
        rgb = np.zeros(shape + (3,), dtype=np.uint8)
        with open_source(source) as reader:
            records = list(_tile_records(source, reader, tiles_root, tile_size, "paired"))
            for tile in records:
                x, y, width, height = tile["x"], tile["y"], tile["valid_width"], tile["valid_height"]
                with Image.open(tile["shadow_path"]) as image:
                    pred[y:y + height, x:x + width] = np.asarray(image)[:height, :width] != 0
                rgb[y:y + height, x:x + width] = reader.read_rgb(x, y, width, height)
                tile["status"] = "complete"
        with (output / "tile_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=TILE_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        _write_png(prediction_path, pred.astype(np.uint8) * 255)
        _write_png(rgb_path, rgb)
        _write_json(cache_path, {"signature": signature, "inference": statistics, "resolved_config": resolved_config})

    with open_source(source) as reader:
        image_valid = reader.read_valid(0, 0, shape[1], shape[0])
        nonzero_input = np.zeros(shape, dtype=bool)
        if source["kind"] == "envi":
            for index in range(reader.header["bands"]):
                nonzero_input |= _band(reader.array, reader.header, index) != 0
        else:
            nonzero_input = np.any(rgb != 0, axis=2)
    valid = image_valid & known
    metrics = evaluate_binary(pred, label, valid)
    sensitivity_valid = valid & nonzero_input
    metrics_nonzero_input = evaluate_binary(pred, label, sensitivity_valid) if sensitivity_valid.any() else None
    reference_png = label.astype(np.uint8) * 255
    reference_png[~valid] = 0
    _write_png(output / "reference_shadow.png", reference_png)
    _write_png(output / "evaluation_valid_mask.png", valid.astype(np.uint8) * 255)
    errors = np.zeros(shape + (3,), dtype=np.uint8)
    errors[valid & pred & label] = (0, 255, 0)
    errors[valid & pred & ~label] = (255, 0, 0)
    errors[valid & ~pred & label] = (0, 0, 255)
    errors[~valid] = (128, 128, 128)
    _write_png(output / "error_map.png", errors)
    _write_jpeg(output / "comparison.jpg", _comparison(rgb, reference_png, pred.astype(np.uint8) * 255, errors))
    report = {
        "metrics": metrics,
        "metrics_nonzero_input": metrics_nonzero_input,
        "metric_units": {"BER": "percent", "FNR": "percent", "FPR": "percent",
                         "other_scores_and_prevalences": "fraction", "TP_TN_FP_FN_N": "pixels"},
        "source": source,
        "reference_path": str(Path(args.reference).resolve()),
        "reference_signature": _signature(args.reference),
        "reference_orientation": args.reference_orientation,
        "reference_values": {"background": 0, "shadow": 255, "other_or_nonfinite": "excluded"},
        "reference_nodata_policy": "Literal zero values are valid background regardless of nodata metadata",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "config_path": str(Path(args.config).resolve()), "resolved_config": resolved_config,
        "prediction_rule": "Argmax of overlapping-averaged class logits; class 1 is shadow",
        "image_geometry": "Unchanged; no resize/rotate/flip of source image",
        "tile_size": tile_size, "stride": stride, "tile_batch_size": batch_size,
        "reused_prediction": reused, "inference": statistics,
        "validity": {"image_valid_pixels": int(np.count_nonzero(image_valid)),
                     "reference_known_pixels": int(np.count_nonzero(known)),
                     "reference_unknown_pixels": int(np.count_nonzero(~known)),
                     "evaluated_pixels": metrics["N"],
                     "all_bands_zero_inside_geometry": int(np.count_nonzero(image_valid & ~nonzero_input)),
                     "nonzero_input_evaluated_pixels": int(np.count_nonzero(sensitivity_valid)),
                     "rule": "Primary: image geometric valid mask AND finite exact binary reference value",
                     "nonzero_input_rule": "Sensitivity only: primary valid AND at least one raw input band nonzero"},
        "error_colors_rgb": {"TN": [0, 0, 0], "TP": [0, 255, 0], "FP": [255, 0, 0],
                             "FN": [0, 0, 255], "invalid_or_unknown": [128, 128, 128]},
        "scope": "One image; not a full dataset benchmark",
        "outputs": {name: str(output / name) for name in (
            "prediction_shadow.png", "reference_shadow.png", "evaluation_valid_mask.png",
            "rgb.png", "error_map.png", "comparison.jpg", "tile_manifest.csv")},
    }
    _write_json(output / "metrics.json", report)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    if metrics_nonzero_input is not None:
        with (output / "metrics_nonzero_input.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics_nonzero_input))
            writer.writeheader()
            writer.writerow(metrics_nonzero_input)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    print(f"Saved report and comparisons: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-orientation", choices=("as-is", "flip-ud"), default="as-is")
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-prediction", action="store_true")
    args = parser.parse_args()
    try:
        run_evaluation(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

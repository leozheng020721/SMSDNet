"""Bounded SEO sample diagnostics; never edits training data or checkpoints."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from model_factory import build_model
from patch_inference import sliding_window_logits


CHECKPOINTS = {
    "nir_minmax": ROOT / "checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_best.pth",
    "older_no_nir": ROOT / "checkpoints/seo_segformer_native512_aux_cca_scratch_mixup_best.pth",
}


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["encoder_weights"] = None
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = {"path": str(path), "epoch_zero_based": checkpoint.get("epoch"),
                "stored_ber": checkpoint.get("ber"), "config": config}
    del checkpoint
    return model.to("cuda").eval(), config, metadata


def read_gray(path):
    with Image.open(path) as source:
        return np.array(source.convert("L"))


def read_sample(sample):
    with Image.open(sample["rgb"]) as source:
        rgb = np.array(source.convert("RGB"))
    minimum, maximum, uncertainty = (read_gray(sample[name]) for name in ("min_mask", "max_mask", "uncertainty"))
    valid = read_gray(sample["nir_valid"]) > 0
    with rasterio.open(sample["nir"]) as source:
        nir = source.read(1, out_dtype="float32")
        valid &= source.read_masks(1) > 0
    valid &= np.isfinite(nir)
    if any(array.shape != rgb.shape[:2] for array in (minimum, maximum, uncertainty, nir, valid)):
        raise ValueError(f"Sample grids differ: {sample['key']}")
    normalized = np.zeros(nir.shape, dtype=np.float32)
    low, high = np.percentile(nir[valid], [1, 99]) if valid.any() else (0, 0)
    if high > low:
        np.subtract(nir, np.float32(low), out=normalized)
        normalized /= np.float32(high - low)
        np.clip(normalized, 0, 1, out=normalized)
        normalized *= np.float32(2)
        normalized -= np.float32(1)
        normalized[~valid] = 0
    return rgb, minimum < 127, maximum < 127, uncertainty == 0, normalized, valid


def predict(model, config, rgb, nir, nir_valid, disable_nir=False):
    normalization = config["dataset"]["normalization"]
    image = np.moveaxis(rgb.astype(np.float32) / 255, -1, 0)
    image -= np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
    image /= np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
    modalities = {}
    if config["dataset"].get("nir", {}).get("enabled", False):
        modalities = {"nir": torch.from_numpy((np.zeros_like(nir) if disable_nir else nir)[None, None]),
                      "nir_valid": torch.from_numpy((np.zeros_like(nir_valid) if disable_nir else nir_valid)[None, None])}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = sliding_window_logits(model, torch.from_numpy(image[None]), "cuda", tile_size=512,
                                       stride=384, tile_batch_size=4, **modalities)
    logits = logits[0].numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite model output")
    return (1 / (1 + np.exp(np.clip(logits[0] - logits[1], -80, 80)))).astype(np.float32)


def metrics(prediction, target, valid):
    positive = target & valid
    negative = ~target & valid
    tp = int((prediction & positive).sum())
    fn = int((~prediction & positive).sum())
    fp = int((prediction & negative).sum())
    tn = int((~prediction & negative).sum())
    miss = fn / (tp + fn) if tp + fn else None
    false_positive = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "ber_percent": 50 * (miss + false_positive) if miss is not None and false_positive is not None else None,
            "shadow_miss_percent": 100 * miss if miss is not None else None,
            "background_false_positive_percent": 100 * false_positive if false_positive is not None else None,
            "shadow_iou": tp / (tp + fn + fp) if tp + fn + fp else None}


def shift_search(prediction, target, valid):
    # A coarse same-sample fit diagnoses direction; it is not a validated correction.
    p, t, v = prediction[::4, ::4], target[::4, ::4], valid[::4, ::4]
    height, width = p.shape
    best, original = None, None
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            shifted = p[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            reference, known = t[y0:y1, x0:x1], v[y0:y1, x0:x1]
            intersection = int((shifted & reference & known).sum())
            union = int(((shifted | reference) & known).sum())
            iou = intersection / union if union else 0
            record = {"dx": dx * 4, "dy": dy * 4, "iou": iou}
            if dx == 0 and dy == 0:
                original = record
            if best is None or iou > best["iou"] + 1e-12 or (
                    abs(iou - best["iou"]) <= 1e-12 and abs(dx) + abs(dy) < (abs(best["dx"]) + abs(best["dy"])) / 4):
                best = record
    return {"best": best, "unshifted": original,
            "note": "Exploratory 4-pixel-grid fit on same sample; positive dx shifts prediction right"}


def outline(binary):
    return binary & ~cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def choose_zoom(maximum, known):
    count, labels, stats, centers = cv2.connectedComponentsWithStats((maximum & known).astype(np.uint8))
    if count > 1:
        index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cx, cy = centers[index]
    else:
        cy, cx = np.asarray(maximum.shape) / 2
    left = max(0, min(maximum.shape[1] - 512, int(cx) - 256))
    top = max(0, min(maximum.shape[0] - 512, int(cy) - 256))
    return left, top


def save_comparison(rgb, minimum, maximum, probabilities, directory, key):
    left, top = choose_zoom(maximum, np.ones_like(maximum))
    cut = (slice(top, top + 512), slice(left, left + 512))
    canvas = Image.new("RGB", (4 * 512, 550), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["Min(cyan) / Max(yellow)", "NIR MinMax prediction(red)",
              "Same weights: NIR disabled(red)", "Older noNIR weights(red): confounded"]
    for index, label in enumerate(labels):
        panel = rgb[cut].copy()
        if index == 0:
            panel[outline(minimum)[cut]] = (0, 255, 255)
            panel[outline(maximum)[cut]] = (255, 255, 0)
        else:
            mode = ("nir_minmax", "nir_disabled", "older_no_nir")[index - 1]
            panel[outline(probabilities[mode] > .5)[cut]] = (255, 0, 0)
        canvas.paste(Image.fromarray(panel), (index * 512, 38))
        draw.text((index * 512 + 8, 7), label, fill="black")
        draw.text((index * 512 + 8, 22), f"{key} | x={left}, y={top}", fill="black")
    canvas.save(directory / "boundary_comparison.png")
    return {"x": left, "y": top, "size": 512,
            "selection": "Centroid of largest Max shadow component"}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((HERE / "sample_manifest.json").read_text(encoding="utf-8"))
    models, configs, checkpoint_metadata = {}, {}, {}
    for name, path in CHECKPOINTS.items():
        print(f"Loading {name}", flush=True)
        models[name], configs[name], checkpoint_metadata[name] = load_model(path)
    results = []
    input_stats = {}
    with rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024):
        for index, sample in enumerate(manifest["samples"], 1):
            print(f"[{index}/{len(manifest['samples'])}] {sample['key']}", flush=True)
            for name in ("rgb", "nir", "nir_valid", "min_mask", "max_mask", "uncertainty"):
                path = Path(sample[name])
                input_stats[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns)
            rgb, minimum, maximum, certain, nir, valid = read_sample(sample)
            agreement = (minimum == maximum) & certain
            probabilities = {}
            for mode in ("nir_minmax", "nir_disabled", "older_no_nir"):
                model_name = "nir_minmax" if mode == "nir_disabled" else mode
                probabilities[mode] = predict(models[model_name], configs[model_name], rgb, nir, valid,
                                               disable_nir=mode == "nir_disabled")
            destination = HERE / "seo_predictions" / sample["key"].replace("/", "__")
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "probabilities.npz", **probabilities)
            result = {"key": sample["key"], "shape": list(maximum.shape),
                      "agreement_valid_fraction": float(agreement.mean()),
                      "nir_valid_fraction": float(valid.mean()), "modes": {}}
            for mode, probability in probabilities.items():
                binary = probability > .5
                Image.fromarray(binary.astype(np.uint8) * 255).save(destination / f"{mode}_mask.png")
                result["modes"][mode] = {"agreement_metrics": metrics(binary, maximum, agreement),
                                         "min_metrics_certain": metrics(binary, minimum, certain),
                                         "max_metrics_certain": metrics(binary, maximum, certain),
                                         "shift_vs_min": shift_search(binary, minimum, certain),
                                         "shift_vs_max": shift_search(binary, maximum, certain)}
            result["zoom"] = save_comparison(rgb, minimum, maximum, probabilities, destination, sample["key"])
            (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            results.append(result)
            print({mode: round(data["agreement_metrics"]["ber_percent"], 3)
                   for mode, data in result["modes"].items()}, flush=True)
    unchanged = all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == value
                    for path, value in input_stats.items())
    if not unchanged:
        raise ValueError("A source file changed during audit")
    aggregate = {}
    for mode in ("nir_minmax", "nir_disabled", "older_no_nir"):
        counts = {key: sum(row["modes"][mode]["agreement_metrics"][key] for row in results)
                  for key in ("tp", "fn", "fp", "tn")}
        tp, fn, fp, tn = (counts[key] for key in ("tp", "fn", "fp", "tn"))
        counts.update(ber_percent=50 * (fn / (tp + fn) + fp / (fp + tn)),
                      shadow_miss_percent=100 * fn / (tp + fn),
                      background_false_positive_percent=100 * fp / (fp + tn),
                      shadow_iou=tp / (tp + fn + fp))
        aggregate[mode] = counts
    report = {"checkpoint_metadata": checkpoint_metadata, "sample_count": len(results),
              "aggregate_agreement_metrics": aggregate, "samples": results, "source_files_unchanged": unchanged,
              "limitations": ["Small diagnostic sample, not full SEO validation or independent manual truth",
                              "NIR disabled is same-checkpoint sensitivity, not retrained ablation",
                              "Older noNIR checkpoint has different training protocol; comparison is confounded",
                              "Shift search fits auto labels on same samples, not a validated geometric correction"]}
    (HERE / "seo_prediction_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "source_files_unchanged": unchanged}, indent=2), flush=True)


if __name__ == "__main__":
    main()

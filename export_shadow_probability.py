"""Export continuous shadow scores using the existing averaged-logit inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def summarize(values):
    values = values[np.isfinite(values)]
    if not values.size:
        return {"pixels": 0}
    quantiles = np.percentile(values, [0, 1, 50, 95, 99, 100])
    return dict(pixels=int(values.size), mean=float(values.mean()),
                **{name: float(value) for name, value in
                   zip(("min", "p01", "median", "p95", "p99", "max"), quantiles)})


def write_previews(probability, rgb, output):
    valid = np.isfinite(probability)
    gray = np.rint(np.nan_to_num(probability, nan=0.0) * 255).astype(np.uint8)
    Image.fromarray(gray).save(output / "shadow_probability_gray.png")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    width = 520
    height = round(rgb.shape[0] * width / rgb.shape[1])
    canvas = Image.new("RGB", (width * 3 + 110, height + 110), "white")
    draw = ImageDraw.Draw(canvas)
    arrays = (rgb, np.repeat(gray[..., None], 3, axis=2),
              cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB))
    arrays[2][~valid] = (200, 200, 200)
    titles = ("Original RGB", "Shadow score: grayscale 0..1", "Shadow score: fixed scale 0..1")
    for index, (array, title) in enumerate(zip(arrays, titles)):
        image = Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(image, (index * width, 65))
        draw.text((index * width + 10, 14), title, font=font, fill="black")
    gradient = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, 25, axis=1)
    colorbar = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(colorbar), (width * 3 + 10, 65))
    for value in (1.0, 0.75, 0.5, 0.25, 0.0):
        y = 65 + round((1.0 - value) * (height - 1))
        draw.text((width * 3 + 40, y - 10), str(value), font=small, fill="black")
    draw.text((10, height + 75), "No threshold applied. Float32 TIFF/NPY preserves the scores. Gray in heatmap = invalid.",
              font=small, fill="black")
    canvas.save(output / "probability_comparison.jpg", quality=95, subsampling=0)
    Image.fromarray(arrays[2]).save(output / "shadow_probability_heatmap.png")


def predict(report_path):
    import torch
    from evaluate_registered_image import read_reference
    from folder_inference import load_model, open_source
    from patch_inference import sliding_window_logits

    cv2.setNumThreads(2)
    report_path = Path(report_path).resolve()
    baseline = json.loads(report_path.read_text(encoding="utf-8"))
    output = report_path.parent / "probability"
    output.mkdir(parents=True, exist_ok=True)
    source = baseline["source"]
    with open_source(source) as reader:
        reader.stretch_summary = baseline["inference"]["stretch"]
        rgb = reader.read_rgb(0, 0, reader.width, reader.height)
        valid = reader.read_valid(0, 0, reader.width, reader.height)
    with Image.open(report_path.parent / "rgb.png") as image:
        if not np.array_equal(rgb, np.asarray(image)):
            raise ValueError("Source RGB differs from the previously evaluated input")
    model, config, device = load_model(baseline["config_path"], baseline["checkpoint_path"])
    mean = np.asarray(config["dataset"]["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(config["dataset"]["normalization"]["std"], dtype=np.float32)
    normalized = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(normalized))[None]
    print("Predicting continuous scores; averaging logits before softmax", flush=True)
    logits = sliding_window_logits(model, inputs, device,
                                   tile_size=baseline["tile_size"], stride=baseline["stride"],
                                   tile_batch_size=baseline["tile_batch_size"])
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model returned non-finite logits")
    probability = logits.softmax(dim=1)[0, 1].numpy().copy()
    label, known = read_reference(baseline["reference_path"], probability.shape,
                                  baseline["reference_orientation"])
    with Image.open(report_path.parent / "prediction_shadow.png") as image:
        previous = np.asarray(image) != 0
    if not np.array_equal((logits.argmax(dim=1)[0].numpy() == 1)[valid], previous[valid]):
        raise RuntimeError("Averaged-logit decisions differ from the previous inference")
    stats = {
        "all_geometrically_valid": summarize(probability[valid]),
        "reference_shadow": summarize(probability[valid & known & label]),
        "reference_background": summarize(probability[valid & known & ~label]),
    }
    probability[~valid] = np.nan
    np.save(output / "shadow_probability.npy", probability)
    metadata = {
        "baseline_report": str(report_path), "source_tif": source["source_tif"],
        "reference_path": baseline["reference_path"],
        "reference_orientation": baseline["reference_orientation"],
        "checkpoint_path": baseline["checkpoint_path"], "shape": list(probability.shape),
        "score": "softmax(mean overlapping class logits)[class 1]",
        "threshold_applied": False, "range": [0.0, 1.0], "invalid": "NaN",
        "input_and_argmax_match_previous_evaluation": True, "statistics": stats,
        "display": "Fixed 0..1 scale; gray PNG quantizes for preview only",
    }
    (output / "probability_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_previews(probability, rgb, output)
    print(json.dumps(stats, indent=2), flush=True)
    print(output, flush=True)


def write_geotiffs(report_path):
    import rasterio

    output = Path(report_path).resolve().parent / "probability"
    metadata = json.loads((output / "probability_report.json").read_text(encoding="utf-8"))
    probability = np.load(output / "shadow_probability.npy")
    for grid, template in (("image_grid", metadata["source_tif"]),
                           ("reference_grid", metadata["reference_path"])):
        with rasterio.open(template) as source:
            profile = source.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate")
        pixels = probability[::-1].copy() if grid == "reference_grid" and metadata["reference_orientation"] == "flip-ud" else probability
        target = output / f"shadow_probability_{grid}.tif"
        with rasterio.open(target, "w", **profile) as destination:
            destination.write(pixels, 1)
            destination.update_tags(score="Shadow class softmax score, 0..1", threshold_applied="false",
                                    fusion="Softmax after overlap-logit averaging", checkpoint=metadata["checkpoint_path"])
        with rasterio.open(target) as check:
            np.testing.assert_array_equal(check.read(1), pixels)
        print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Existing single-image metrics.json")
    parser.add_argument("--write-geotiffs-only", action="store_true")
    args = parser.parse_args()
    if args.write_geotiffs_only:
        write_geotiffs(args.report)
    else:
        predict(args.report)

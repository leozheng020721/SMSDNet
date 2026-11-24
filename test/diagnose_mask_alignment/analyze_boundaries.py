"""Compare local image edges and prediction boundaries without shifting data."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parent
SOURCE = Path(r"E:\Dataset\21century\registration_output\rectified\4\region_01\image_1_rectified.tif")
PREDICTIONS = ROOT / "test/results_century_nir_minmax/4/region_01"
WINDOW = Window(5120, 3072, 1024, 1024)


def edge_measurements(probability, gray, roi):
    """Brightness edges are a diagnostic reference, not annotated truth."""
    anchor, rows, bounds = roi
    _, labels = cv2.connectedComponents((probability > 0.5).astype(np.uint8))
    identity = labels[anchor[1], anchor[0]]
    if identity == 0:
        return {"available": False, "reason": "Anchor is not predicted shadow"}
    derivative = np.diff(gray, axis=1)
    measurements = {"left": [], "right": []}
    lbound, rbound = bounds
    for y in range(*rows):
        positions = np.flatnonzero(labels[y, lbound:rbound] == identity) + lbound
        if positions.size < 10:
            continue
        for side, predicted in (("left", positions[0]), ("right", positions[-1])):
            left, right = max(1, predicted - 25), min(gray.shape[1] - 2, predicted + 25)
            gradient = derivative[y, left:right]
            offset = np.argmin(gradient) if side == "left" else np.argmax(gradient)
            reference = left + int(offset) + 0.5
            measurements[side].append({"row": y, "prediction_x": int(predicted),
                                       "brightness_edge_x": reference,
                                       "prediction_minus_brightness_edge": float(predicted - reference)})
    result = {"available": True, "rows": list(rows), "x_bounds": list(bounds)}
    for side, records in measurements.items():
        values = [record["prediction_minus_brightness_edge"] for record in records]
        result[side] = {"count": len(values), "median_pixels": float(np.median(values)) if values else None,
                        "p10_p90_pixels": np.percentile(values, [10, 90]).tolist() if values else None,
                        "records": records}
    return result


def contour(ax, mask, color, label=None):
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0)
    if label:
        ax.plot([], [], color=color, label=label)


def main():
    with rasterio.open(SOURCE) as source:
        data = source.read(window=WINDOW)
    with rasterio.open(PREDICTIONS / "image_1_shadow_probability.tif") as source:
        saved = source.read(1, window=WINDOW)
    with rasterio.open(PREDICTIONS / "image_1_shadow_mask.tif") as source:
        mask = source.read(1, window=WINDOW) > 0
    rgb = np.moveaxis(np.clip(data[[2, 1, 0]].astype(np.float32) / 4095, 0, 1), 0, -1)
    nir = np.clip((data[3].astype(np.float32) - 367) / (4095 - 367), 0, 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (7, 1), 1.2)
    modes = {"saved": saved}
    arrays_path = DESTINATION / "sensitivity_arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path) as arrays:
            modes.update({name: arrays[name] for name in ("real", "no_nir", "hflip")})
    regions = {"upper_shadow": ((500, 350), (330, 381), (400, 690)),
               "long_shadow": ((800, 420), (412, 436), (585, 1015))}
    report = {"source": str(SOURCE), "crop": [5120, 3072, 1024, 1024],
              "reference": "Strongest signed horizontal RGB brightness edge within 25 pixels; not manual ground truth",
              "positive_offset": "Prediction lies to the right of brightness edge",
              "measurements": {mode: {name: edge_measurements(p, gray_smooth, roi)
                                      for name, roi in regions.items()} for mode, p in modes.items()}}
    (DESTINATION / "boundary_measurements.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for ax, image, title in ((axes[0, 0], rgb, "RGB + saved mask boundary"),
                             (axes[0, 1], nir, "NIR + same saved boundary")):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
        contour(ax, mask, "red")
        ax.set(xlim=(585, 1015), ylim=(480, 365), title=title,
               xlabel="Local image x (pixels)", ylabel="Local image y (pixels)")
    ys = slice(416, 430)
    colors = {"saved": "red", "real": "red", "no_nir": "cyan", "hflip": "magenta"}
    for ax, bounds, title in ((axes[1, 0], (590, 690), "Long shadow: LEFT edge"),
                              (axes[1, 1], (940, 1015), "Long shadow: RIGHT edge")):
        left, right = bounds
        xx = np.arange(left, right)
        for field, color, label in ((gray, "black", "RGB brightness (scaled)"),
                                    (nir, "green", "NIR brightness (scaled)")):
            profile = field[ys, left:right].mean(axis=0)
            low, high = np.percentile(profile, [5, 95])
            profile = np.clip((profile - low) / max(float(high - low), 1e-6), 0, 1)
            ax.plot(xx, profile, color=color, linewidth=2, label=label)
        for mode, probabilities in modes.items():
            if mode == "real":
                continue
            ax.plot(xx, probabilities[ys, left:right].mean(axis=0), color=colors[mode],
                    linewidth=1.5, label=f"P(shadow): {mode}")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set(xlim=bounds, ylim=(-0.02, 1.02), title=title,
               xlabel="Local image x (pixels)", ylabel="Relative brightness / shadow probability")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Image 1 / tile x5120 y3072 | No coordinate shifts applied | Brightness edges are not ground truth")
    fig.savefig(DESTINATION / "edge_diagnostics.png", dpi=130)
    plt.close(fig)

    if len(modes) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
        for ax, mode, title in zip(axes, ("real", "no_nir", "hflip"),
                                  ("Real NIR (reproduced)", "NIR disabled (same checkpoint)",
                                   "Horizontal flip, then flip output back")):
            ax.imshow(rgb)
            contour(ax, modes[mode] > 0.5, "red")
            ax.set(xlim=(400, 1024), ylim=(650, 270), title=title,
                   xlabel="Local image x (pixels)", ylabel="Local image y (pixels)")
        fig.suptitle("Input sensitivity diagnostic; no retraining or coordinate correction")
        fig.savefig(DESTINATION / "sensitivity_comparison.png", dpi=130)
        plt.close(fig)
    for mode, measurements in report["measurements"].items():
        print(mode, {name: {side: data.get(side, {}).get("median_pixels") for side in ("left", "right")}
                     for name, data in measurements.items()}, flush=True)
    print(f"Saved diagnostics: {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()

"""Summarize measured evidence without treating auto labels as human truth."""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent


def read_rgb(path):
    with Image.open(path) as source:
        return np.array(source.convert("RGB"))


def read_mask(path):
    with Image.open(path) as source:
        return np.array(source.convert("L"))


def main():
    manifest = json.loads((HERE / "sample_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "seo_prediction_audit.json").read_text(encoding="utf-8"))
    sample = next(row for row in manifest["samples"] if row["key"] == "JAX_231/JAX_231_24")
    rgb = read_rgb(sample["rgb"])
    minimum, maximum = (read_mask(sample[name]) < 127 for name in ("min_mask", "max_mask"))
    certain = read_mask(sample["uncertainty"]) == 0
    known = (minimum == maximum) & certain
    with np.load(HERE / "seo_predictions/JAX_231__JAX_231_24/probabilities.npz") as arrays:
        predictions = {name: arrays[name] for name in arrays.files}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y, left, right = 400, 330, 420
    values = {"strong_dark_gray_lt40": gray < 40, "min": minimum, "max": maximum,
              **{name: probability > .5 for name, probability in predictions.items()}}
    first_edges = {}
    for name, binary in values.items():
        positions = np.flatnonzero(binary[y, left:right])
        runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        main_run = max(runs, key=len)
        first_edges[name] = int(left + main_run[0]) if main_run.size else None
    summary = {"sample_count": audit["sample_count"],
               "aggregate_agreement_metrics": audit["aggregate_agreement_metrics"],
               "local_boundary_example": {"key": sample["key"], "y": y, "x_search": [left, right],
                                          "main_shadow_start_x": first_edges,
                                          "edge_selection": "Start of longest contiguous shadow run in x_search",
                                          "uncertainty": "Brightness threshold is a visible-edge diagnostic, not manually annotated ground truth"},
               "max_shadow_exclusion_fraction_range": [min(row["stats"]["max_shadow_excluded_fraction"] for row in manifest["samples"]),
                                                       max(row["stats"]["max_shadow_excluded_fraction"] for row in manifest["samples"])],
               "limits": audit["limitations"], "source_files_unchanged": audit["source_files_unchanged"]}
    (HERE / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.8]},
                             constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].contour(minimum, levels=[.5], colors=["cyan"], linewidths=1)
    axes[0].contour(maximum, levels=[.5], colors=["yellow"], linewidths=1)
    axes[0].contour(predictions["nir_minmax"], levels=[.5], colors=["red"], linewidths=1)
    axes[0].axhline(y, color="white", linestyle="--", linewidth=1)
    axes[0].set(xlim=(320, 420), ylim=(460, 340), title="RGB + Min(cyan), Max(yellow), prediction(red)",
                xlabel="Original image x", ylabel="Original image y")
    x = np.arange(left, right)
    axes[1].plot(x, gray[y, left:right] / 255, color="black", label="RGB brightness / 255")
    axes[1].step(x, minimum[y, left:right], color="cyan", where="mid", label="Min shadow label")
    axes[1].step(x, maximum[y, left:right], color="goldenrod", where="mid", label="Max shadow label")
    axes[1].plot(x, predictions["nir_minmax"][y, left:right], color="red", label="P(shadow): NIR enabled")
    axes[1].plot(x, predictions["nir_disabled"][y, left:right], color="magenta", label="P(shadow): NIR disabled")
    axes[1].fill_between(x, 0, 1, where=~known[y, left:right], color="gray", alpha=.15,
                         label="Ignored by training supervision", step="mid")
    axes[1].axhline(.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set(xlim=(left, right), ylim=(-.02, 1.02), title="Scanline y=400: labels and probability",
                xlabel="Original image x", ylabel="Brightness / label / probability")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    fig.suptitle("Held-out JAX_231_24 | Boundary floats inside Min/Max disagreement band; not manual ground truth")
    fig.savefig(HERE / "jax231_boundary_evidence.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

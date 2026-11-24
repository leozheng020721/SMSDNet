"""Select deterministic held-out SEO scenes and audit label geometry on CPU."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupShuffleSplit


BASE = Path(__file__).resolve().parent
ROOT = Path(r"E:\Dataset\SEO")
SOURCE = BASE.parent / "results_century_nir_minmax/4/region_01/image_1_prediction.json"


def load_mask(path):
    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise ValueError(f"Cannot read {path}")
    return data


def sample_stats(paths):
    minimum = load_mask(paths["min_mask"])
    maximum = load_mask(paths["max_mask"])
    uncertainty = load_mask(paths["uncertainty"])
    valid = load_mask(paths["nir_valid"]) > 0
    if not all(data.shape == maximum.shape for data in (minimum, uncertainty, valid)):
        raise ValueError(f"Label shape mismatch for {paths['key']}")
    disagreement = minimum != maximum
    ignored = disagreement | (uncertainty != 0)
    supervised = ~ignored
    shadow = maximum < 127
    return {
        "width": int(maximum.shape[1]), "height": int(maximum.shape[0]),
        "nir_coverage": float(valid.mean()),
        "min_shadow_fraction": float((minimum < 127).mean()),
        "max_shadow_fraction": float(shadow.mean()),
        "minmax_disagreement_fraction": float(disagreement.mean()),
        "min_uncertainty_fraction": float((uncertainty != 0).mean()),
        "supervised_fraction": float(supervised.mean()),
        "supervised_shadow_fraction": float(shadow[supervised].mean()) if supervised.any() else None,
        "max_shadow_excluded_fraction": float(ignored[shadow].mean()) if shadow.any() else None,
        "max_shadow_excluded_by_disagreement_fraction": float(disagreement[shadow].mean()) if shadow.any() else None,
        "max_shadow_excluded_by_uncertainty_fraction": float((uncertainty[shadow] != 0).mean()) if shadow.any() else None,
        "max_shadow_disagreement_only_fraction": float((disagreement & (uncertainty == 0))[shadow].mean()) if shadow.any() else None,
        "max_shadow_uncertainty_only_fraction": float((~disagreement & (uncertainty != 0))[shadow].mean()) if shadow.any() else None,
        "max_shadow_both_exclusion_fraction": float((disagreement & (uncertainty != 0))[shadow].mean()) if shadow.any() else None,
        "min_only_shadow_fraction": float(((minimum < 127) & ~shadow).mean()),
        "max_only_shadow_fraction": float(((minimum >= 127) & shadow).mean()),
        "supervised_shadow_pixels": int((shadow & supervised).sum()),
        "max_shadow_pixels": int(shadow.sum()),
        "mask_values": {"min": np.unique(minimum).tolist(), "max": np.unique(maximum).tolist(),
                        "uncertainty": np.unique(uncertainty).tolist()},
    }


def main():
    if "--audit-existing" in sys.argv:
        target = BASE / "sample_manifest.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        for item in payload["samples"]:
            item["stats"] = sample_stats(item)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        geometry_audit(payload["samples"])
        detail_crops(payload["samples"])
        print("Existing sample statistics updated.", flush=True)
        return
    config = json.loads(SOURCE.read_text(encoding="utf-8"))["checkpoint"]["config"]
    max_root = ROOT / config["dataset"]["supervision"]["max_root"]
    min_root = ROOT / config["dataset"]["supervision"]["min_root"]
    pairs = []
    for image in sorted((ROOT / "pansharpened_he").rglob("*_rgb.png")):
        mask = max_root / "shadow_masks" / image.parent.name / image.name.replace("_rgb.png", "_shadow.png")
        if mask.is_file():
            pairs.append((image, mask, image.parent.name))
    groups = [item[2] for item in pairs]
    splitter = GroupShuffleSplit(n_splits=1, test_size=config["dataset"].get("val_ratio", .2),
                                random_state=config["random_seed"])
    train_indices, val_indices = next(splitter.split(pairs, groups=groups))
    manifest = ROOT / "nir_aligned_v1/registration_with_candidates.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        eligible = {row["key"]: row for row in csv.DictReader(handle)
                    if row["reference_grid"] == "RGB" and row["status"] in {"original_rgb_ready", "rgb_ready"}}
    candidates = {city: [] for city in ("JAX", "OMA", "UCSD")}
    seen_groups = {city: set() for city in candidates}
    eligible_val = 0
    complete_val = 0
    for index in val_indices:
        image, maximum, group = pairs[index]
        key = group + "/" + image.name.removesuffix("_rgb.png")
        row = eligible.get(key)
        if row is None:
            continue
        eligible_val += 1
        minimum = min_root / "shadow_masks" / group / maximum.name
        uncertainty = min_root / "uncertainty_masks" / group / maximum.name.replace("_shadow.png", "_uncertainty_mask.png")
        if not (minimum.is_file() and uncertainty.is_file()):
            continue
        complete_val += 1
        city = group.split("_")[0]
        if city not in candidates or group in seen_groups[city] or len(candidates[city]) >= 12:
            continue
        if float(row.get("valid_fraction") or 0) < .90:
            continue
        paths = {"key": key, "city": city, "aoi": group, "pair_index": int(index), "split": "val",
                 "rgb": str(image), "nir": row["nir"], "nir_valid": row["valid"],
                 "min_mask": str(minimum), "max_mask": str(maximum), "uncertainty": str(uncertainty),
                 "registration_status": row["status"]}
        stats = sample_stats(paths)
        paths.update(width=stats["width"], height=stats["height"], stats=stats)
        if stats["nir_coverage"] < .9 or stats["max_shadow_fraction"] < .015:
            continue
        score = (abs(stats["max_shadow_fraction"] - .15) + stats["min_uncertainty_fraction"]
                 + .25 * stats["minmax_disagreement_fraction"])
        paths["selection_score"] = score
        candidates[city].append(paths)
        seen_groups[city].add(group)
    samples = []
    for city in candidates:
        selected = sorted(candidates[city], key=lambda item: (item["selection_score"], item["key"]))[:2]
        samples.extend(selected)
    payload = {
        "purpose": "Small deterministic held-out diagnostic; not an unbiased dataset-wide benchmark.",
        "split": {"method": "GroupShuffleSplit", "random_seed": config["random_seed"],
                  "val_ratio": config["dataset"].get("val_ratio", .2), "all_pairs": len(pairs),
                  "train_pairs_before_nir_filter": len(train_indices), "val_pairs_before_nir_filter": len(val_indices),
                  "val_after_nir_filter": eligible_val, "val_complete_supervision": complete_val},
        "selection": "First up to 12 sorted validation AOIs per city with >=90% NIR coverage; choose two closest to 15% shadow with low uncertainty.",
        "label_convention": "Source 0=shadow, 255=nonshadow. Training target 1=shadow; ignore Min/Max disagreement or nonzero Min uncertainty.",
        "vegetation_masks": {
            "local_root_directories": [path.name for path in ROOT.iterdir() if path.is_dir()],
            "min_directories": [path.name for path in min_root.iterdir() if path.is_dir()],
            "max_directories": [path.name for path in max_root.iterdir() if path.is_dir()],
            "observation": "No separately named vegetation-mask directory in local SEO root or Min/Max label roots. No vegetation masks are loaded by dataset.py.",
        },
        "samples": samples,
    }
    target = BASE / "sample_manifest.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(target), flush=True)
    print(json.dumps({"samples": [{"key": item["key"], "rgb": item["rgb"], "stats": item["stats"]} for item in samples]}, indent=2), flush=True)
    geometry_audit(samples)
    detail_crops(samples)


def geometry_audit(samples):
    panels = []
    evidence = []
    kernel = np.ones((3, 3), np.uint8)
    for item in samples:
        rgb = np.asarray(Image.open(item["rgb"]).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 50, 120) > 0
        distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        minimum = load_mask(item["min_mask"]) < 127
        maximum = load_mask(item["max_mask"]) < 127
        uncertainty = load_mask(item["uncertainty"]) != 0
        record = {"key": item["key"], "caution": "RGB luminance edges are not human shadow ground truth; nearby building/vegetation texture can bias these distances."}
        for name, shadow in (("min", minimum), ("max", maximum)):
            outline = (shadow.astype(np.uint8) - cv2.erode(shadow.astype(np.uint8), kernel)) > 0
            outline[:2] = outline[-2:] = False
            outline[:, :2] = outline[:, -2:] = False
            measure = distance[outline & ~uncertainty]
            record[name + "_boundary_to_rgb_edge"] = {
                "boundary_pixels_excluding_uncertainty": int(measure.size),
                "median_px": float(np.median(measure)) if measure.size else None,
                "p90_px": float(np.percentile(measure, 90)) if measure.size else None,
                "fraction_within_3px": float((measure <= 3).mean()) if measure.size else None,
            }
        evidence.append(record)
        variants = []
        for title, shadow in (("RGB", None), ("Min: red boundary", minimum), ("Max: red; ignored: yellow", maximum)):
            display = rgb.copy()
            if shadow is not None:
                outline = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
                display[outline] = (255, 0, 0)
            if title.startswith("Max"):
                ignored = (minimum != maximum) | uncertainty
                display[ignored] = (.60 * display[ignored] + .40 * np.array([255, 230, 0])).astype(np.uint8)
            thumb = Image.fromarray(display)
            thumb.thumbnail((470, 390))
            panel = Image.new("RGB", (480, 430), "white")
            panel.paste(thumb, ((480 - thumb.width) // 2, 34))
            draw = ImageDraw.Draw(panel)
            draw.text((6, 5), item["key"] + " | " + title, fill="black")
            variants.append(panel)
        row = Image.new("RGB", (1440, 430), "white")
        for index, panel in enumerate(variants):
            row.paste(panel, (480 * index, 0))
        panels.append(row)
    sheet = Image.new("RGB", (1440, 430 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, 430 * index))
    sheet.save(BASE / "label_rgb_contact_sheet.jpg", quality=93)
    (BASE / "label_geometry_audit.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Geometry audit and contact sheet complete.", flush=True)


def detail_crops(samples):
    chosen = {"JAX_255/JAX_255_24": (500, 150, 850, 800),
              "JAX_231/JAX_231_24": (250, 250, 950, 750)}
    rows = []
    metadata = []
    kernel = np.ones((3, 3), np.uint8)
    for item in samples:
        if item["key"] not in chosen:
            continue
        x, y, width, height = chosen[item["key"]]
        crop = np.s_[y:y + height, x:x + width]
        rgb = np.asarray(Image.open(item["rgb"]).convert("RGB"))[crop]
        minimum = load_mask(item["min_mask"])[crop] < 127
        maximum = load_mask(item["max_mask"])[crop] < 127
        uncertainty = load_mask(item["uncertainty"])[crop] != 0
        ignored = (minimum != maximum) | uncertainty
        joint = rgb.copy()
        min_edge = cv2.morphologyEx(minimum.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        max_edge = cv2.morphologyEx(maximum.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        joint[min_edge] = (255, 0, 0)
        joint[max_edge] = (0, 255, 255)
        training = rgb.copy()
        training[ignored] = (.45 * training[ignored] + .55 * np.array([255, 230, 0])).astype(np.uint8)
        training[maximum & ~ignored] = (.45 * training[maximum & ~ignored] + .55 * np.array([255, 0, 0])).astype(np.uint8)
        panel_width = 800
        panel_height = 800
        row = Image.new("RGB", (panel_width * 3, panel_height), "white")
        for i, (title, data) in enumerate((("RGB", rgb), ("Min red / Max cyan", joint),
                                          ("Training: shadow red / ignored yellow", training))):
            image = Image.fromarray(data)
            image.thumbnail((panel_width - 10, panel_height - 50))
            row.paste(image, (i * panel_width + 5, 45))
            ImageDraw.Draw(row).text((i * panel_width + 5, 5), item["key"] + " | " + title, fill="black")
        rows.append(row)
        metadata.append({"key": item["key"], "x": x, "y": y, "width": width, "height": height,
                         "note": "Coordinates refer to original RGB pixels, not resized contact-sheet pixels. Visible brightness is not manual ground truth."})
    sheet = Image.new("RGB", (2400, 800 * len(rows)), "white")
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * 800))
    sheet.save(BASE / "label_detail_crops.jpg", quality=95)
    (BASE / "label_detail_crops.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

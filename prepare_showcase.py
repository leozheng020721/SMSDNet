"""Export curated, traceable display examples without running a model."""

import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from folder_inference import discover_sources, open_source


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jpg":
        image.convert("RGB").save(path, quality=95, subsampling=0)
    else:
        image.save(path)


def overview(cases, title, path):
    columns = 3 if len(cases) >= 5 else 2
    rows = (len(cases) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 536 + 24, rows * 580 + 96), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), title, fill="#17212b", font=font(31))
    for index, case in enumerate(cases):
        x, y = 24 + (index % columns) * 536, 76 + (index // columns) * 580
        with Image.open(case["display_path"]) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
            canvas.paste(thumbnail, (x, y))
        draw.text((x, y + 518), f"{case['id']}  {case['title']}", fill="#17212b", font=font(21))
    write_image(path, canvas)


def comparison(rgb, mask, marked, case):
    canvas = Image.new("RGB", (1536, 564), "white")
    draw = ImageDraw.Draw(canvas)
    labels = case.get("labels", ("Original RGB", "Predicted shadow", "Shadow outlines"))
    for x, label, image in zip((0, 512, 1024), labels, (rgb, mask, marked)):
        draw.text((x + 12, 8), label, fill="#17212b", font=font(22))
        canvas.paste(image.convert("RGB"), (x, 44))
    return canvas


def export_cases(selection_path, output, update_mountains=False):
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not update_mountains:
        raise ValueError(f"Use a new or empty output folder: {output}")
    existing_city = []
    if update_mountains:
        with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            existing_city = [row for row in csv.DictReader(stream) if row["category"] == "city_good"]
    output.mkdir(parents=True, exist_ok=True)
    sources, readers = {}, {}
    rows = []
    with ExitStack() as stack:
        for category in (("mountain_issues",) if update_mountains else ("mountain_issues", "city_good")):
            for case in selection[category]:
                source_root = str(Path(case["source_root"]).resolve())
                if source_root not in sources:
                    sources[source_root] = discover_sources(source_root)
                record = next(s for s in sources[source_root] if s["stem"] == case["stem"]
                              and s["relative_parent"] == case.get("relative_parent", "."))
                source_key = record["source_path"]
                if source_key not in readers:
                    readers[source_key] = stack.enter_context(open_source(record))
                reader = readers[source_key]
                if "stretch" in case:
                    reader.stretch_summary = case["stretch"]
                x, y = int(case["x"]), int(case["y"])
                width, height = int(case.get("width", 512)), int(case.get("height", 512))
                if min(width, height) <= 0 or x < 0 or y < 0 or x + width > reader.width or y + height > reader.height:
                    raise ValueError(f"Case must contain its original pixels without padding: {case['id']}")
                if category == "city_good" and (width, height) != (512, 512):
                    raise ValueError("City comparisons require the existing 512 x 512 predictions")
                valid = reader.read_valid(x, y, width, height)
                if not valid.all():
                    raise ValueError(f"Select a fully valid showcase crop: {case['id']}")
                rgb = Image.fromarray(reader.read_rgb(x, y, width, height))
                case.update(width=width, height=height, valid_fraction=float(valid.mean()),
                            black_fraction=float(np.mean(np.max(np.asarray(rgb), axis=-1) == 0)))
                target = output / category
                rgb_path = target / f"{case['id']}_rgb.png"
                write_image(rgb_path, rgb)
                row = {
                    "id": case["id"], "category": category, "title": case["title"],
                    "reason": case["reason"], "source_path": record["source_path"],
                    "source_tif": record.get("source_tif", ""), "x": x, "y": y,
                    "width": width, "height": height, "rgb_path": str(rgb_path),
                    "mask_path": "", "marked_path": "", "comparison_path": "",
                    "inference_performed": category == "city_good",
                    "stretch": json.dumps(reader.stretch_summary, ensure_ascii=False),
                }
                if category == "city_good":
                    mask_path = target / f"{case['id']}_mask.png"
                    marked_path = target / f"{case['id']}_marked.png"
                    comparison_path = target / f"{case['id']}_comparison.png"
                    with Image.open(case["mask_path"]) as image:
                        mask = image.copy()
                    with Image.open(case["marked_path"]) as image:
                        marked = image.copy()
                    if mask.size != (512, 512) or marked.size != (512, 512):
                        raise ValueError(f"Prediction size differs: {case['id']}")
                    marked_array = np.asarray(marked)
                    red = np.all(marked_array == [255, 0, 0], axis=-1)
                    if not np.array_equal(np.asarray(rgb)[~red], marked_array[~red]):
                        raise ValueError(f"RGB display differs from inferred source: {case['id']}")
                    write_image(mask_path, mask)
                    write_image(marked_path, marked)
                    write_image(comparison_path, comparison(rgb, mask, marked, case))
                    row.update(mask_path=str(mask_path), marked_path=str(marked_path), comparison_path=str(comparison_path))
                row["display_path"] = row["marked_path"] or row["rgb_path"]
                rows.append(row)
    rows.extend(existing_city)
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    for category, title in (("mountain_issues", "Mountain image issues | Original RGB crops"), ("city_good", "City predictions | Visually selected examples")):
        if update_mountains and category == "city_good":
            continue
        cases = [r for r in rows if r["category"] == category]
        overview(cases, selection.get(category + "_title", title), output / f"{category}_overview.jpg")
    notes = [
        "# Showcase examples", "",
        "Mountain crops show observed image appearance issues; no mountain inference was run.",
        "City cases are visually selected predictions, not a quantitative accuracy evaluation.",
        "Individual crops preserve the source pixel dimensions recorded in manifest.csv.",
        "City masks: white = predicted shadow. Red lines mark predicted shadow boundaries.",
        "The 3-column comparison PNGs contain RGB, prediction mask, and marked RGB.",
        "16-bit source RGB is displayed with a fixed per-source stretch recorded in manifest.csv.",
        "Source files and existing predictions were not modified.", "",
        "| ID | Case | Observation |", "| --- | --- | --- |",
    ]
    notes.extend(f"| {row['id']} | {row['title']} | {row['reason']} |" for row in rows)
    if not update_mountains:
        (output / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(rows), "mountain": len(selection["mountain_issues"]), "city": len(selection["city_good"])}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--update-mountains", action="store_true", help="Replace mountain crops and metadata while retaining city files")
    args = parser.parse_args()
    export_cases(args.selection, args.output, update_mountains=args.update_mountains)

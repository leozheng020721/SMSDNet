"""Windowed ENVI/RGB inference and non-overlapping square PNG exports."""

import argparse
import csv
import json
import math
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


ENVI_DTYPES = {
    1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
    12: "u2", 13: "u4", 14: "i8", 15: "u8",
}
TILE_FIELDS = (
    "source_path", "source_width", "source_height", "relative_parent", "stem",
    "tile_id",
    "x", "y", "width", "height", "valid_width", "valid_height", "pad_right", "pad_bottom",
    "valid_pixels", "valid_fraction", "rgb_path", "shadow_path", "overlay_path", "marked_path", "status",
)
SOURCE_FIELDS = (
    "source_path", "data_path", "valid_header", "valid_data", "source_tif",
    "relative_parent", "stem", "width", "height", "bands", "dtype", "rgb_indices",
    "total_tiles", "valid_tiles", "empty_tiles", "total_pixels", "valid_pixels", "selected",
)


def parse_envi_header(path):
    path = Path(path).resolve()
    content = path.read_text(encoding="utf-8-sig")
    fields = {}
    for match in re.finditer(r"(?m)^\s*([^=\r\n]+?)\s*=\s*(\{[^}]*\}|[^\r\n]*)", content):
        key = " ".join(match.group(1).lower().split())
        value = match.group(2).strip()
        fields[key] = value[1:-1].strip() if value.startswith("{") else value
    for name in ("samples", "lines", "bands", "data type"):
        if name not in fields:
            raise ValueError(f"Missing ENVI field {name}: {path}")
        fields[name] = int(fields[name])
    if min(fields["samples"], fields["lines"], fields["bands"]) <= 0:
        raise ValueError(f"ENVI dimensions must be positive: {path}")
    code = fields["data type"]
    if code not in ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type {code}: {path}")
    order = int(fields.get("byte order", 0))
    if order not in (0, 1):
        raise ValueError(f"Unsupported ENVI byte order: {path}")
    fields["numpy_dtype"] = np.dtype(ENVI_DTYPES[code]).newbyteorder("<" if order == 0 else ">")
    fields["header offset"] = int(fields.get("header offset", 0))
    if fields["header offset"] < 0:
        raise ValueError(f"ENVI header offset must be nonnegative: {path}")
    fields["interleave"] = fields.get("interleave", "bip").lower()
    if fields["interleave"] not in ("bip", "bil", "bsq"):
        raise ValueError(f"Unsupported ENVI interleave: {path}")
    fields["band names"] = [s.strip() for s in fields.get("band names", "").split(",") if s.strip()]
    fields["default bands"] = [int(s.strip()) - 1 for s in fields.get("default bands", "").split(",") if s.strip()]
    explicit = fields.get("data file")
    candidates = [path.parent / explicit] if explicit else [path.with_suffix(".dat"), path.with_suffix("")]
    data_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(f"ENVI binary file not found for {path}")
    expected = fields["header offset"] + fields["samples"] * fields["lines"] * fields["bands"] * fields["numpy_dtype"].itemsize
    if data_path.stat().st_size < expected:
        raise ValueError(f"ENVI binary is shorter than its header dimensions: {data_path}")
    fields.update(header_path=str(path), data_path=str(data_path), width=fields["samples"], height=fields["lines"])
    return fields


def _rgb_indices(header):
    names = [name.strip().upper() for name in header["band names"]]
    if all(color in names for color in ("RED", "GREEN", "BLUE")):
        return [names.index(color) for color in ("RED", "GREEN", "BLUE")]
    default = header["default bands"]
    if len(default) >= 3 and all(0 <= index < header["bands"] for index in default[:3]):
        return default[:3]
    if header["bands"] == 3:
        return [0, 1, 2]
    raise ValueError(f"Cannot identify RGB bands from ENVI header: {header['header_path']}")


def _open_map(header):
    height, width, bands = header["height"], header["width"], header["bands"]
    shapes = {"bip": (height, width, bands), "bil": (height, bands, width), "bsq": (bands, height, width)}
    return np.memmap(header["data_path"], mode="r", dtype=header["numpy_dtype"],
                     offset=header["header offset"], shape=shapes[header["interleave"]])


def _band(array, header, index):
    if header["interleave"] == "bip":
        return array[:, :, index]
    if header["interleave"] == "bil":
        return array[:, index, :]
    return array[index]


def _close_map(array):
    if array is not None:
        array._mmap.close()


class EnviReader:
    def __init__(self, source):
        self.source = source
        self.header = parse_envi_header(source["source_path"])
        self.width, self.height = self.header["width"], self.header["height"]
        self.rgb_indices = source["rgb_indices"] if "rgb_indices" in source else _rgb_indices(self.header)
        self.array = _open_map(self.header)
        self.valid_header = None
        self.valid_array = None
        valid_path = source.get("valid_header")
        if valid_path:
            self.valid_header = parse_envi_header(valid_path)
            if (self.valid_header["width"], self.valid_header["height"], self.valid_header["bands"]) != (self.width, self.height, 1):
                self.close()
                raise ValueError(f"Valid mask dimensions differ from image: {valid_path}")
            self.valid_array = _open_map(self.valid_header)
        self.stretch_summary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        _close_map(self.valid_array)
        _close_map(self.array)
        self.valid_array = self.array = None

    def _check_window(self, x, y, width, height):
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > self.width or y + height > self.height:
            raise ValueError("Window must lie within the original raster")

    def read_valid(self, x, y, width, height):
        self._check_window(x, y, width, height)
        if self.valid_array is None:
            return np.ones((height, width), dtype=bool)
        return _band(self.valid_array, self.valid_header, 0)[y:y + height, x:x + width] != 0

    def _prepare_stretch(self):
        dtype = self.header["numpy_dtype"]
        if dtype.kind == "u" and dtype.itemsize == 1:
            self.stretch_summary = {"method": "uint8_identity", "low": [0.0] * 3, "high": [255.0] * 3}
            return
        if dtype.kind not in ("u", "i") or dtype.itemsize != 2:
            raise ValueError("Automatic percentile RGB conversion is restricted to 16-bit integer ENVI images")
        step = max(1, math.ceil(math.sqrt(self.width * self.height / 262144)))
        valid = (
            _band(self.valid_array, self.valid_header, 0)[::step, ::step] != 0
            if self.valid_array is not None else np.ones((math.ceil(self.height / step), math.ceil(self.width / step)), dtype=bool)
        )
        samples = [np.asarray(_band(self.array, self.header, index)[::step, ::step][valid]) for index in self.rgb_indices]
        # Very small valid islands may fall between the deterministic sample grid.
        if not samples[0].size and self.valid_array is not None:
            for top in range(0, self.height, 128):
                mask = self.read_valid(0, top, self.width, min(128, self.height - top))
                rows, columns = np.nonzero(mask)
                if rows.size:
                    rows, columns = rows[:262144] + top, columns[:262144]
                    samples = [np.asarray(_band(self.array, self.header, index)[rows, columns]) for index in self.rgb_indices]
                    break
        low, high = [], []
        for sample in samples:
            a, b = np.percentile(sample, [2, 98]) if sample.size else (0.0, 1.0)
            low.append(float(a))
            high.append(float(b) if b > a else float(a) + 1.0)
        self.stretch_summary = {
            "method": "valid_global_sample_percentile_2_98", "sample_step": step,
            "sample_count": int(samples[0].size), "rgb_indices": self.rgb_indices,
            "low": low, "high": high,
        }

    def read_rgb(self, x, y, width, height):
        self._check_window(x, y, width, height)
        if self.stretch_summary is None:
            self._prepare_stretch()
        raw = np.stack([_band(self.array, self.header, index)[y:y + height, x:x + width]
                        for index in self.rgb_indices], axis=-1).astype(np.float32)
        low = np.asarray(self.stretch_summary["low"], dtype=np.float32)
        high = np.asarray(self.stretch_summary["high"], dtype=np.float32)
        rgb = np.clip((raw - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        rgb[~self.read_valid(x, y, width, height)] = 0
        return rgb


class RGBReader:
    def __init__(self, source):
        self.source = source
        self.width, self.height = source["width"], source["height"]
        self.image = None
        self.stretch_summary = {"method": "uint8_identity", "low": [0.0] * 3, "high": [255.0] * 3}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.image is not None:
            self.image.close()

    def read_valid(self, x, y, width, height):
        return np.ones((height, width), dtype=bool)

    def read_rgb(self, x, y, width, height):
        if self.image is None:
            self.image = Image.open(self.source["source_path"]).convert("RGB")
        return np.asarray(self.image.crop((x, y, x + width, y + height)), dtype=np.uint8).copy()


def discover_sources(input_root):
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    headers = sorted(root.rglob("*.hdr"), key=lambda p: str(p.relative_to(root)).casefold())
    candidates = [p for p in headers if "valid" not in p.stem.lower() and "mask" not in p.stem.lower()]
    sources = []
    for path in candidates:
        info = parse_envi_header(path)
        if info["bands"] < 3:
            continue
        stem = re.sub(r"_envi$", "", path.stem, flags=re.IGNORECASE)
        valid_path = path.with_name(stem + "_valid_mask.hdr")
        valid = parse_envi_header(valid_path) if valid_path.is_file() else None
        tif_path = next((path.with_name(stem + suffix) for suffix in (".tif", ".tiff")
                         if path.with_name(stem + suffix).is_file()), None)
        sources.append({
            "kind": "envi", "source_path": str(path.resolve()), "data_path": info["data_path"],
            "valid_header": str(valid_path.resolve()) if valid else "", "valid_data": valid["data_path"] if valid else "",
            "source_tif": str(tif_path.resolve()) if tif_path else "", "relative_parent": str(path.parent.relative_to(root)),
            "stem": stem, "width": info["width"], "height": info["height"], "bands": info["bands"],
            "dtype": str(info["numpy_dtype"]), "rgb_indices": _rgb_indices(info),
        })
    envi_keys = {(source["relative_parent"].casefold(), source["stem"].casefold()) for source in sources}
    priority = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4, ".bmp": 5}
    unique = {}
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root)).casefold()):
        if not path.is_file() or path.suffix.lower() not in priority:
            continue
        if re.search(r"(?:preview|overlay|valid.?mask|_pred$|_shadow$|_vis$|(?:^|[_-])pair(?:[_-]|$))", path.stem, re.IGNORECASE):
            continue
        key = (str(path.parent.relative_to(root)).casefold(), path.stem.casefold())
        if key in envi_keys:
            continue
        if key not in unique or priority[path.suffix.lower()] < priority[unique[key].suffix.lower()]:
            unique[key] = path
    for path in sorted(unique.values(), key=lambda p: str(p.relative_to(root)).casefold()):
        with Image.open(path) as image:
            if image.mode not in ("RGB", "RGBA", "L", "P"):
                raise ValueError(f"Use its ENVI header for non-uint8 raster input: {path}")
            width, height = image.size
        sources.append({
            "kind": "rgb", "source_path": str(path.resolve()), "data_path": str(path.resolve()),
            "valid_header": "", "valid_data": "", "source_tif": str(path.resolve()) if path.suffix.lower() in (".tif", ".tiff") else "",
            "relative_parent": str(path.parent.relative_to(root)), "stem": path.stem,
            "width": width, "height": height, "bands": 3, "dtype": "uint8", "rgb_indices": [0, 1, 2],
        })
    if not sources:
        raise ValueError(f"No supported ENVI or RGB images found in {root}")
    return sorted(sources, key=lambda source: (source["relative_parent"].casefold(), source["stem"].casefold()))


def open_source(source):
    return EnviReader(source) if source["kind"] == "envi" else RGBReader(source)


def tile_starts(length, tile_size=512, stride=384):
    if min(length, tile_size, stride) <= 0 or stride > tile_size:
        raise ValueError("Require positive dimensions and 0 < stride <= tile_size")
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _validate_export_layout(export_layout):
    if export_layout not in ("triplet", "paired"):
        raise ValueError("export_layout must be 'triplet' or 'paired'")


def _tile_records(source, reader, output_root, tile_size, export_layout="triplet"):
    _validate_export_layout(export_layout)
    target = Path(output_root) / source["relative_parent"]
    for y in range(0, reader.height, tile_size):
        for x in range(0, reader.width, tile_size):
            width, height = min(tile_size, reader.width - x), min(tile_size, reader.height - y)
            valid_pixels = int(np.count_nonzero(reader.read_valid(x, y, width, height)))
            tile_id = f"{source['stem']}_r{y:05d}_c{x:05d}"
            prefix = target / tile_id
            if export_layout == "paired":
                filename = tile_id + ".png"
                paths = {
                    "rgb_path": "", "overlay_path": "",
                    "shadow_path": str(Path(output_root) / "masks" / source["relative_parent"] / filename),
                    "marked_path": str(Path(output_root) / "marked" / source["relative_parent"] / filename),
                }
            else:
                paths = {
                    "rgb_path": str(prefix) + "_rgb.png", "shadow_path": str(prefix) + "_shadow.png",
                    "overlay_path": str(prefix) + "_overlay.png", "marked_path": "",
                }
            yield {
                "source_path": source["source_path"], "source_width": reader.width, "source_height": reader.height,
                "relative_parent": source["relative_parent"], "stem": source["stem"], "tile_id": tile_id,
                "x": x, "y": y,
                "width": tile_size, "height": tile_size, "valid_width": width, "valid_height": height,
                "pad_right": tile_size - width, "pad_bottom": tile_size - height,
                "valid_pixels": valid_pixels, "valid_fraction": valid_pixels / (width * height),
                **paths, "status": "planned",
            }


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_source_manifest(path, sources):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sources)


def scan_folder(input_root, output_root, tile_size=512, count=None, export_layout="triplet"):
    if tile_size <= 0 or (count is not None and count <= 0):
        raise ValueError("tile_size and count must be positive")
    _validate_export_layout(export_layout)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(input_root)
    selected_count = len(sources) if count is None else min(count, len(sources))
    tile_manifest = output / "tile_manifest.csv"
    with tile_manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=TILE_FIELDS)
        writer.writeheader()
        for index, source in enumerate(sources):
            source.update(total_tiles=0, valid_tiles=0, empty_tiles=0, total_pixels=source["width"] * source["height"], valid_pixels=0, selected=index < selected_count)
            with open_source(source) as reader:
                for tile in _tile_records(source, reader, output, tile_size, export_layout):
                    source["total_tiles"] += 1
                    source["valid_tiles"] += tile["valid_pixels"] > 0
                    source["empty_tiles"] += tile["valid_pixels"] == 0
                    source["valid_pixels"] += tile["valid_pixels"]
                    if source["selected"]:
                        writer.writerow(tile)
    selected = sources[:selected_count]
    summary = {"source_count": len(sources), "selected_source_count": selected_count,
               "tile_size": tile_size, "export_layout": export_layout}
    for field in ("total_tiles", "valid_tiles", "empty_tiles", "total_pixels", "valid_pixels"):
        summary[field] = sum(source[field] for source in sources)
        summary["selected_" + field] = sum(source[field] for source in selected)
    summary["fully_invalid_tiles"] = summary["empty_tiles"]
    summary["selected_fully_invalid_tiles"] = summary["selected_empty_tiles"]
    source_manifest = output / "source_manifest.csv"
    _write_source_manifest(source_manifest, sources)
    result = {"summary": summary, "sources": sources, "selected_sources": selected,
              "source_manifest": str(source_manifest), "tile_manifest": str(tile_manifest)}
    _write_json(output / "scan_summary.json", result)
    return result


def load_model(config_path, checkpoint_path, device=None):
    # Keep torch and model imports out of the scan-only path.
    import copy
    import torch
    import yaml
    from model_factory import build_model, resolve_checkpoint_config

    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = resolve_checkpoint_config(config, checkpoint)
    selected_device = torch.device(device or config.get("device", "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    model_config = copy.deepcopy(config)
    model_config["model"]["encoder_weights"] = None
    model = build_model(model_config)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model = model.to(selected_device).eval()
    return model, config, selected_device


def _write_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(".png", data)
    if not ok:
        raise OSError(f"Unable to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def draw_shadow_outline(rgb, shadow):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Outline image must be uint8 HxWx3 RGB")
    if shadow.ndim != 2 or shadow.shape != rgb.shape[:2]:
        raise ValueError("Shadow mask must match the RGB image height and width")
    binary = np.ascontiguousarray((shadow != 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked = rgb.copy()
    cv2.drawContours(marked, contours, -1, (255, 0, 0), thickness=1)
    return marked


def _predict_source(model, config, device, source, output, tile_size, stride, tile_batch_size,
                    export_layout="triplet"):
    import torch

    _validate_export_layout(export_layout)
    normalization = config["dataset"]["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("RGB normalization needs three means and positive standard deviations")
    classes = int(config["dataset"].get("num_classes", 2))
    if classes < 2:
        raise ValueError("Expected at least two class logits, with shadow at class 1")
    statistics = {"source_path": source["source_path"], "inference_windows": 0, "model_windows": 0, "skipped_windows": 0, "exported_tiles": 0}
    with open_source(source) as reader, tempfile.TemporaryDirectory(prefix=".inference-", dir=output) as work:
        sums = coverage = None
        try:
            sums = np.memmap(Path(work) / "logits.dat", mode="w+", dtype=np.float32, shape=(classes, reader.height, reader.width))
            coverage = np.memmap(Path(work) / "coverage.dat", mode="w+", dtype=np.uint32, shape=(reader.height, reader.width))
            patches, positions = [], []

            def flush():
                if not patches:
                    return
                batch = np.stack(patches).astype(np.float32) / 255.0
                batch = np.ascontiguousarray(((batch - mean) / std).transpose(0, 3, 1, 2))
                with torch.inference_mode():
                    logits = model(torch.from_numpy(batch).to(device))
                    if isinstance(logits, dict):
                        logits = logits["out"]
                    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (len(patches), classes, tile_size, tile_size):
                        raise ValueError("Model must return tile-sized BxClassesxHxW logits")
                    logits = logits.detach().float().cpu().numpy()
                for index, (x, y, width, height) in enumerate(positions):
                    sums[:, y:y + height, x:x + width] += logits[index, :, :height, :width]
                    coverage[y:y + height, x:x + width] += 1
                patches.clear()
                positions.clear()

            row_starts = tile_starts(reader.height, tile_size, stride)
            column_starts = tile_starts(reader.width, tile_size, stride)
            windows = ((x, y) for y in row_starts for x in column_starts)
            for x, y in tqdm(windows, total=len(row_starts) * len(column_starts), desc="Inference windows", unit="window"):
                statistics["inference_windows"] += 1
                width, height = min(tile_size, reader.width - x), min(tile_size, reader.height - y)
                valid = reader.read_valid(x, y, width, height)
                if not valid.any():
                    statistics["skipped_windows"] += 1
                    continue
                patch = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                patch[:height, :width] = reader.read_rgb(x, y, width, height)
                patches.append(patch)
                positions.append((x, y, width, height))
                statistics["model_windows"] += 1
                if len(patches) == tile_batch_size:
                    flush()
            flush()
            exports = _tile_records(source, reader, output, tile_size, export_layout)
            for tile in tqdm(exports, total=source["total_tiles"], desc="Export tiles", unit="tile"):
                x, y, width, height = tile["x"], tile["y"], tile["valid_width"], tile["valid_height"]
                valid = reader.read_valid(x, y, width, height)
                counts = np.asarray(coverage[y:y + height, x:x + width])
                if np.any(valid & (counts == 0)):
                    raise RuntimeError("A valid image pixel was not covered by inference windows")
                averaged = np.array(sums[:, y:y + height, x:x + width], copy=True)
                averaged /= np.maximum(counts, 1).astype(np.float32)[None]
                predicted = (averaged.argmax(axis=0) == 1) & valid
                rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                rgb[:height, :width] = reader.read_rgb(x, y, width, height)
                shadow = np.zeros((tile_size, tile_size), dtype=np.uint8)
                shadow[:height, :width] = predicted.astype(np.uint8) * 255
                _write_png(tile["shadow_path"], shadow)
                if export_layout == "paired":
                    marked = draw_shadow_outline(rgb, shadow)
                    padded_valid = np.zeros((tile_size, tile_size), dtype=bool)
                    padded_valid[:height, :width] = valid
                    marked[~padded_valid] = 0
                    _write_png(tile["marked_path"], marked)
                else:
                    overlay = rgb.copy()
                    selected = shadow != 0
                    overlay[selected] = (0.5 * overlay[selected].astype(np.float32) + np.array([127.5, 0, 0], dtype=np.float32)).astype(np.uint8)
                    _write_png(tile["rgb_path"], rgb)
                    _write_png(tile["overlay_path"], overlay)
                statistics["exported_tiles"] += 1
            statistics["stretch"] = reader.stretch_summary
        finally:
            _close_map(coverage)
            _close_map(sums)
    return statistics


def run_inference(args):
    cv2.setNumThreads(2)
    tile_size = int(getattr(args, "tile_size", 512))
    export_layout = getattr(args, "export_layout", "triplet")
    result = scan_folder(args.input, args.output, tile_size=tile_size,
                         count=getattr(args, "count", None), export_layout=export_layout)
    summary = result["summary"]
    print(f"Sources: {summary['source_count']} | All grid tiles: {summary['total_tiles']} | Valid tiles: {summary['valid_tiles']} | Empty tiles: {summary['empty_tiles']}", flush=True)
    print(f"Selected sources: {summary['selected_source_count']} | Selected grid tiles: {summary['selected_total_tiles']}", flush=True)
    if getattr(args, "scan_only", False):
        return result
    if not getattr(args, "config", None) or not getattr(args, "checkpoint", None):
        raise ValueError("Inference requires --config and --checkpoint")
    stride = int(getattr(args, "stride", 384))
    tile_batch_size = int(getattr(args, "tile_batch_size", 2))
    if not 0 < stride <= tile_size or tile_batch_size <= 0:
        raise ValueError("Require 0 < stride <= tile_size and a positive tile_batch_size")
    model, config, device = load_model(args.config, args.checkpoint, getattr(args, "device", None))
    model.eval()
    result["inference"] = []
    output = Path(args.output).resolve()
    for index, source in enumerate(result["selected_sources"], start=1):
        print(f"[{index}/{summary['selected_source_count']}] {source['relative_parent']}/{source['stem']}: {source['width']}x{source['height']}, {source['total_tiles']} export tiles", flush=True)
        statistics = _predict_source(model, config, device, source, output, tile_size, stride,
                                     tile_batch_size, export_layout)
        result["inference"].append(statistics)
        print(f"Exported {statistics['exported_tiles']} tiles; model windows {statistics['model_windows']}, skipped {statistics['skipped_windows']}", flush=True)
    # The manifest retains the original coordinates and distinguishes completed exports.
    with Path(result["tile_manifest"]).open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    with Path(result["tile_manifest"]).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=TILE_FIELDS)
        writer.writeheader()
        for record in records:
            record["status"] = "complete"
            writer.writerow(record)
    result["config_path"], result["checkpoint_path"] = str(Path(args.config).resolve()), str(Path(args.checkpoint).resolve())
    result["stride"], result["tile_batch_size"] = stride, tile_batch_size
    _write_json(output / "inference_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=None, help="Limit original images, not exported tiles")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=384, help="Overlapping inference stride; export stride stays tile-size")
    parser.add_argument("--export-layout", choices=("triplet", "paired"), default="triplet",
                        help="triplet: RGB/mask/filled overlay; paired: masks/marked folders with matching names")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    try:
        run_inference(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

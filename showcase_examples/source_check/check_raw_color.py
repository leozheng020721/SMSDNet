"""Audit original MS/PAN and fused data in the mountain showcase footprints."""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = Path(r"E:\BaiduNetdiskDownload\21ATRD25148-朱艳丽-0805\阴影相关数据")
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def percentiles(values):
    if not values.size:
        return None
    return dict(zip(("p01", "p10", "median", "p90", "p99"),
                    np.percentile(values, [1, 10, 50, 90, 99]).tolist()))


def band_stats(data, mask):
    if not mask.any():
        return {"count": 0}
    values = data[:, mask].astype(np.float64)
    return {
        "count": int(mask.sum()),
        "BGRN_medians": np.median(values, axis=1).tolist(),
        "B_div_R": percentiles(values[0] / np.maximum(values[2], 1)),
        "B_div_G": percentiles(values[0] / np.maximum(values[1], 1)),
        "red_zero_fraction": float(np.mean(values[2] == 0)),
    }


def render(data, high):
    rgb = data[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    return np.rint(np.clip(rgb / max(high, 1), 0, 1) * 255).astype(np.uint8)


def raster_info(ds):
    return {
        "path": ds.name, "size": [ds.width, ds.height], "bands": ds.count,
        "dtype": ds.dtypes, "tags": ds.tags(), "scales": ds.scales,
        "offsets": ds.offsets, "size_bytes": Path(ds.name).stat().st_size,
    }


def draw_panels(images, labels, notes, path):
    side, top, footer = 560, 74, 90
    canvas = Image.new("RGB", (side * len(images), side + top + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    for i, (array, label) in enumerate(zip(images, labels)):
        draw.text((i * side + 12, 12), label, fill="black", font=font)
        image = Image.fromarray(array)
        # Nearest enlargement of the native MS preserves its original pixels.
        image.thumbnail((side, side), Image.Resampling.NEAREST)
        if max(image.size) < side:
            factor = side / max(image.size)
            image = image.resize((round(image.width * factor), round(image.height * factor)),
                                 Image.Resampling.NEAREST)
        canvas.paste(image, (i * side + (side - image.width) // 2,
                             top + (side - image.height) // 2))
    for i, line in enumerate(notes):
        draw.text((12, top + side + 8 + 25 * i), line, fill="black", font=small)
    canvas.save(path, quality=95)


def main():
    spec = importlib.util.spec_from_file_location("fusion_audit", RAW / "pansharpen_batch.py")
    fusion = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fusion
    spec.loader.exec_module(fusion)
    selection = json.loads((WORKSPACE / "showcase_examples/selection.json").read_text(encoding="utf-8"))
    report = {
        "scope": "Original group 7; exact inverse-homography footprints of M01/M02, both dates",
        "output": str(OUT),
        "display": "Raw MS and fused shared: RGB bands 3,2,1; same scalar DN range [0,p98(raw visible)], gamma=1. PAN grayscale has its own p98. Final fused display has its own scalar p98.",
        "blue_definition": "Native or cubic-upsampled B > 1.5*G and B > 1.5*R and B > 10 DN; valid footprint only. This is a descriptive selection, not calibrated reflectance or a land-cover label.",
        "limitations": "No physical calibration coefficients are provided. DN channel ratios and blue appearance establish that the effect predates fusion; they do not establish sensor fault or the physical cause.",
        "cases": [],
    }
    for item in selection["mountain_issues"]:
        if item["id"] not in {"M01", "M02"}:
            continue
        region = Path(item["source_root"])
        meta = json.loads((region / "registration.json").read_text(encoding="utf-8"))
        for scene_idx, fused_path in enumerate((meta["image_a"], meta["image_b"])):
            key = f'{item["id"]}_scene{scene_idx + 1}'
            scene = Path(fused_path).parent.name
            raw_scene = RAW / "7" / scene
            ms_path = next(p for p in (raw_scene / "MS").glob("*.tif")
                           if not p.stem.endswith("_mask"))
            pan_path = next(p for p in (raw_scene / "PAN").glob("*.tif")
                            if not p.stem.endswith("_mask"))
            matrix = np.asarray(meta["homographies_source_to_rectangle"][scene_idx])
            x, y, width, height = (item[k] for k in ("x", "y", "width", "height"))
            corners = np.float64([[[x, y], [x + width, y],
                                  [x + width, y + height], [x, y + height]]])
            source_corners = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))[0]
            lo = (np.floor(source_corners.min(axis=0) / 4) * 4).astype(int)
            hi = (np.ceil(source_corners.max(axis=0) / 4) * 4).astype(int)
            with rasterio.open(ms_path) as ms, rasterio.open(pan_path) as pan, rasterio.open(fused_path) as fused_ds:
                if pan.width != ms.width * 4 or pan.height != ms.height * 4:
                    raise ValueError("MS/PAN scale is not exactly four")
                lo = np.maximum(lo, 0)
                hi = np.minimum(hi, [pan.width, pan.height])
                win = Window(*lo, *(hi - lo))
                native_win = Window(*(lo // 4), *((hi - lo) // 4))
                native = ms.read(window=native_win)
                upsampled = fusion.read_ms_on_pan_grid(ms, win, 4)
                pan_dn = pan.read(1, window=win)
                fused = fused_ds.read(window=win)
                local_poly = source_corners - lo
                inside = np.zeros(pan_dn.shape, np.uint8)
                cv2.fillPoly(inside, [np.rint(local_poly).astype(np.int32)], 1)
                valid = (inside > 0) & (fused_ds.dataset_mask(window=win) > 0)
                valid &= np.all(upsampled > 0, axis=0)
                inside_native = np.zeros(native.shape[1:], np.uint8)
                cv2.fillPoly(inside_native, [np.rint(local_poly / 4).astype(np.int32)], 1)
                native_valid = (inside_native > 0) & np.all(native > 0, axis=0)
                native_valid &= fused_ds.dataset_mask(window=win,
                    out_shape=native.shape[1:], resampling=Resampling.nearest) > 0
                native_blue = native_valid & (native[0] > native[1].astype(float) * 1.5)
                native_blue &= (native[0] > native[2].astype(float) * 1.5) & (native[0] > 10)
                blue = valid & (upsampled[0] > upsampled[1].astype(float) * 1.5)
                blue &= (upsampled[0] > upsampled[2].astype(float) * 1.5) & (upsampled[0] > 10)
                intensity = upsampled.astype(np.float32).mean(axis=0)
                gain = pan_dn / np.maximum(intensity, 1)
                dark = valid & (intensity <= np.percentile(intensity[valid], 25))
                expected = fusion.weighted_brovey(upsampled, pan_dn, valid,
                    np.ones(4, np.float32) / 4,
                    int(fused_ds.tags().get("OUTPUT_MAX_VALUE", "4095")), 8)
                shared_high = float(np.percentile(native[:3, native_valid], 98))
                fused_high = float(np.percentile(fused[:3, valid], 98))
                pan_high = float(np.percentile(pan_dn[valid], 98))
                native_rgb = render(native, shared_high)
                fused_rgb = render(fused, shared_high)
                fused_auto = render(fused, fused_high)
                pan_gray = np.rint(np.clip(pan_dn / max(pan_high, 1), 0, 1) * 255).astype(np.uint8)
                Image.fromarray(native_rgb).save(OUT / f"{key}_native_ms_rgb.png")
                Image.fromarray(pan_gray).save(OUT / f"{key}_pan_gray.png")
                Image.fromarray(fused_rgb).save(OUT / f"{key}_fused_shared.png")
                Image.fromarray(fused_auto).save(OUT / f"{key}_fused_display.png")
                display_arrays = [native_rgb.copy(), np.repeat(pan_gray[..., None], 3, axis=2),
                                  fused_rgb.copy(), fused_auto.copy()]
                for i, data in enumerate(display_arrays):
                    polygon = local_poly / 4 if i == 0 else local_poly
                    cv2.polylines(data, [np.rint(polygon).astype(np.int32)], True, (255, 220, 0), 1 if i == 0 else 3)
                date = "2024-12-31" if scene_idx == 0 else "2025-12-10"
                draw_panels(display_arrays,
                    ["原始多光谱 RGB（无融合）", "原始全色 PAN（单通道）",
                     "融合 RGB（与左图同 DN 标尺）", "融合 RGB（独立亮度显示）"],
                    [f"{item['id']}  /  {date}  /  黄色边框为 1024×1024 展示区域在原始图中的范围；外框是读取的原图矩形窗口。",
                     "原始 MS 按真实原生像素读取并最近邻放大；RGB 各通道共用同一线性标尺，未分别调色。",
                     "DN 不是已校准反射率；该图用于判断偏蓝是否在融合前已存在，不直接判定传感器故障。"],
                    OUT / f"{key}_comparison.jpg")
                case = {
                    "id": key, "scene": scene, "showcase_rect": [x, y, width, height],
                    "source_window_xywh": [int(v) for v in (win.col_off, win.row_off, win.width, win.height)],
                    "source_footprint_xy": source_corners.tolist(),
                    "native_ms_window_xywh": [int(v) for v in (native_win.col_off, native_win.row_off, native_win.width, native_win.height)],
                    "original_ms": raster_info(ms), "original_pan": raster_info(pan),
                    "fused": raster_info(fused_ds),
                    "native_blue_fraction": float(native_blue.sum() / native_valid.sum()),
                    "native_valid_count": int(native_valid.sum()),
                    "pan_grid_valid_count": int(valid.sum()),
                    "pan_grid_footprint_valid_fraction": float(valid.sum() / np.count_nonzero(inside)),
                    "native_blue": band_stats(native, native_blue),
                    "native_nonblue": band_stats(native, native_valid & ~native_blue),
                    "upsampled_blue": band_stats(upsampled, blue),
                    "fused_at_same_blue_pixels": band_stats(fused, blue),
                    "gain_blue": percentiles(gain[blue]),
                    "gain_all": percentiles(gain[valid]),
                    "gain_dark_quartile": percentiles(gain[dark]),
                    "pan_at_blue": percentiles(pan_dn[blue]),
                    "pan_at_dark_quartile": percentiles(pan_dn[dark]),
                    "expected_brovey_exact_fraction": float(np.mean(np.all(fused[:, valid] == expected[:, valid], axis=0))),
                    "saturated_any_band_fraction": float(np.mean(np.any(fused[:, valid] >= 4095, axis=0))),
                    "dark_fused_visible_all_at_most5_fraction": float(np.mean(np.max(fused[:3, dark], axis=0) <= 5)),
                    "display_high_common_ms_and_fused": shared_high,
                    "display_high_fused_independent": fused_high,
                    "display_high_pan": pan_high,
                    "comparison": str(OUT / f"{key}_comparison.jpg"),
                }
                report["cases"].append(case)
                print(json.dumps({"id": key, "native_blue_fraction": case["native_blue_fraction"],
                    "native_blue_BGRN_median": case["native_blue"].get("BGRN_medians"),
                    "native_nonblue_BGRN_median": case["native_nonblue"].get("BGRN_medians"),
                    "fused_same_blue_BGRN_median": case["fused_at_same_blue_pixels"].get("BGRN_medians"),
                    "gain_blue": case["gain_blue"],
                    "formula_agreement": case["expected_brovey_exact_fraction"]}), flush=True)
    (OUT / "raw_color_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Read-only RPC/PAN geometry probe; outputs are restricted to this workspace."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("GDAL_CACHEMAX", "32")

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import RPCTransformer
from rasterio.warp import transform

from align_seo_nir import stretch_rgb


REPOSITORY = Path(__file__).resolve().parent
EXAMPLES = ("JAX_436_8", "JAX_486_15", "OMA_930_26", "UCSD_291_3")


def apply_h(points, h):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[:, None], h)[:, 0]


def rpc_map(source_rpc, pan_rpc, points, height):
    """Use native center indices; rasterio rowcol returns corner-based indices."""
    heights = np.full(len(points), height, np.float64)
    with RPCTransformer(source_rpc, RPC_PIXEL_ERROR_THRESHOLD=0.001) as camera:
        lon, lat = camera.xy(points[:, 1], points[:, 0], zs=heights, offset="center")
        rows, cols = camera.rowcol(lon, lat, zs=heights, op=lambda value: value)
    back = np.column_stack((cols, rows)) - 0.5
    with RPCTransformer(pan_rpc, RPC_PIXEL_ERROR_THRESHOLD=0.001) as camera:
        rows, cols = camera.rowcol(lon, lat, zs=heights, op=lambda value: value)
    return (np.column_stack((cols, rows)) - 0.5, np.column_stack((lon, lat)),
            float(np.linalg.norm(back - points, axis=1).max()))


def manual_rpc_projection(rpc, lonlat, height):
    """Independent RPC00B polynomial check, without GDAL's forward transformer."""
    lon = (lonlat[:, 0] - rpc.long_off) / rpc.long_scale
    lat = (lonlat[:, 1] - rpc.lat_off) / rpc.lat_scale
    z = np.full(len(lon), (height - rpc.height_off) / rpc.height_scale)
    terms = np.array((np.ones(len(lon)), lon, lat, z, lon * lat, lon * z,
                      lat * z, lon ** 2, lat ** 2, z ** 2, lon * lat * z,
                      lon ** 3, lon * lat ** 2, lon * z ** 2, lon ** 2 * lat,
                      lat ** 3, lat * z ** 2, lon ** 2 * z, lat ** 2 * z, z ** 3))
    x = rpc.samp_off + rpc.samp_scale * (
        np.asarray(rpc.samp_num_coeff) @ terms) / (np.asarray(rpc.samp_den_coeff) @ terms)
    y = rpc.line_off + rpc.line_scale * (
        np.asarray(rpc.line_num_coeff) @ terms) / (np.asarray(rpc.line_den_coeff) @ terms)
    return np.column_stack((x, y))


def dsm_info(root, aoi):
    paths = sorted((root / "rdsm_max_compressed" / aoi).glob("*.tif"))
    if not paths:
        return None
    with rasterio.open(paths[0]) as dataset:
        values = dataset.read(1, masked=True).compressed()
        values = values[np.isfinite(values)]
        x, y = dataset.transform * (dataset.width / 2, dataset.height / 2)
        lon, lat = transform(dataset.crs, "EPSG:4326", [x], [y])
    return {"path": str(paths[0]), "median_height_m": float(np.median(values)),
            "p5_p95_height_m": np.percentile(values, (5, 95)).tolist(),
            "center_lonlat": [lon[0], lat[0]],
            "height_caution": "DSM CRS declares ellipsoidal height; median is a plane approximation"}


def read_cached_candidates(path, keys):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("key") not in keys:
                continue
            passed = [candidate for candidate in row.get("candidates", [])
                      if candidate.get("qa_passed") and candidate.get("H_native_to_pan")]
            if passed:
                result[row["key"]] = max(passed, key=lambda item: item["mutual_matches"])
    return result


def panel(image, size=(320, 340)):
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return Image.fromarray(image).resize(size)


def preview(path, source, pan, target, h, reference_h):
    width, height = pan.shape[1], pan.shape[0]
    aligned = cv2.warpPerspective(source, h, (width, height))
    pan_rgb = np.repeat(pan[:, :, None], 3, axis=2)
    edges_pan = cv2.Canny(pan, 50, 120)
    edges_rpc = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), 50, 120)
    edges = np.stack((edges_rpc, edges_pan, edges_pan), axis=-1)
    images = [pan_rgb, target, source, aligned, edges]
    labels = ["Original PAN", "Current RGB", "Native MSI visible",
              "RPC-aligned visible", "PAN cyan / RPC red edges"]
    if reference_h is not None:
        images.append(cv2.warpPerspective(source, reference_h, (width, height)))
        labels.append("Cached image-match alignment")
    canvas = Image.new("RGB", (320 * len(images), 370), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(panel(image), (index * 320, 30))
        draw.text((index * 320 + 4, 8), label, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\Dataset\SEO"))
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY / "nir_registration_benchmark" / "rpc_probe_v1")
    parser.add_argument("--cache", type=Path,
                        default=REPOSITORY / "nir_registration_benchmark" / "comparison_v1" / "pan_candidates.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    if REPOSITORY != output and REPOSITORY not in output.parents:
        raise ValueError("Probe outputs must remain inside the repository workspace")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    keys = {"_".join(stem.split("_")[:2]) + "/" + stem for stem in EXAMPLES}
    cached = read_cached_candidates(args.cache, keys)
    records = []
    for stem in EXAMPLES:
        aoi = "_".join(stem.split("_")[:2])
        key = aoi + "/" + stem
        with rasterio.open(args.root / "msi" / aoi / (stem + "_msi.tif")) as dataset:
            source_rpc = dataset.rpcs
            source_tags = dataset.tags()
            source = stretch_rgb(np.moveaxis(dataset.read([3, 2, 1]), 0, -1))
        with rasterio.open(args.root / "crops" / aoi / (stem + "_pan.tif")) as dataset:
            pan_rpc = dataset.rpcs
            pan_tags = dataset.tags()
            pan = dataset.read(1).astype(np.float32)
        target = np.asarray(Image.open(args.root / "pansharpened_he" / aoi /
                                      (stem + "_rgb.png")).convert("RGB"))
        sh, sw = source.shape[:2]
        ph, pw = pan.shape
        points = np.float64([(x, y) for y in np.linspace(2, sh - 3, 13)
                             for x in np.linspace(2, sw - 3, 13)])
        dsm = dsm_info(args.root, aoi)
        heights = [source_rpc.height_off - 100, source_rpc.height_off,
                   source_rpc.height_off + 100]
        if dsm:
            heights.append(dsm["median_height_m"])
        row = {"key": key, "source_size_wh": [sw, sh], "pan_size_wh": [pw, ph],
               "same_source_id": source_tags.get("NITF_PIAIMC_SOURCE") == pan_tags.get("NITF_PIAIMC_SOURCE"),
               "source_id": source_tags.get("NITF_PIAIMC_SOURCE"),
               "source_acquisition": source_tags.get("NITF_STDIDC_ACQUISITION_DATE"),
               "pan_acquisition": pan_tags.get("NITF_STDIDC_ACQUISITION_DATE"),
               "dsm": dsm, "planes": [],
               "caution": "RPC-grid fit and roundtrip are model self-consistency checks, not image registration validation"}
        reference = cached.get(key)
        reference_h = np.asarray(reference["H_native_to_pan"], np.float64) if reference else None
        if stem == "JAX_486_15":
            # Previously recorded diagnostic fit; retained only for visual comparison.
            reference_h = np.array([[3.85862844, -0.05461169, -19.11128819],
                                    [-0.00016379, 3.81998731, -1304.06943002],
                                    [-0.000006275, -0.000089549, 1]], np.float64)
        for height in heights:
            projected, lonlat, roundtrip = rpc_map(source_rpc, pan_rpc, points, height)
            h, _ = cv2.findHomography(points, projected, 0)
            independent = manual_rpc_projection(pan_rpc, lonlat, height)
            fits = apply_h(points, h)
            within = ((projected[:, 0] >= 0) & (projected[:, 0] < pw)
                      & (projected[:, 1] >= 0) & (projected[:, 1] < ph))
            case = {"height_m": height, "H_rpc_plane": h.tolist(),
                    "rpc_roundtrip_max_source_px": roundtrip,
                    "manual_polynomial_difference_max_pan_px": float(np.linalg.norm(independent - projected, axis=1).max()),
                    "rpc_plane_fit_max_pan_px": float(np.linalg.norm(fits - projected, axis=1).max()),
                    "projected_grid_inside_pan_fraction": float(within.mean()),
                    "source_center_lonlat": lonlat[len(points) // 2].tolist()}
            if reference is not None:
                src = np.asarray(reference["points_src_native"], np.float64)
                dst = np.asarray(reference["points_dst_pan"], np.float64)
                visual_error = np.linalg.norm(apply_h(src, reference_h) - dst, axis=1)
                keep = visual_error < 6
                error = np.linalg.norm(apply_h(src[keep], h) - dst[keep], axis=1)
                case["image_correspondence_comparison"] = {
                    "strategy": reference["strategy"], "points": int(keep.sum()),
                    "reference_passes_old_random_QA_only": True,
                    "rpc_median_pan_px": float(np.median(error)),
                    "rpc_p90_pan_px": float(np.percentile(error, 90)),
                    "reference_median_pan_px": float(np.median(visual_error[keep]))}
            row["planes"].append(case)
        base_h = np.asarray(row["planes"][1]["H_rpc_plane"])
        baseline = apply_h(points, base_h)
        row["height_sensitivity_max_pan_px"] = max(
            float(np.linalg.norm(apply_h(points, np.asarray(case["H_rpc_plane"])) - baseline, axis=1).max())
            for case in row["planes"])
        if reference_h is not None:
            row["image_H_reference"] = reference_h.tolist()
            delta = np.linalg.norm(apply_h(points, reference_h) - baseline, axis=1)
            row["RPC_vs_image_H_grid_median_pan_px"] = float(np.median(delta))
        low, high = np.percentile(pan, (1, 99))
        displayed_pan = np.uint8(np.clip((pan - low) / max(high - low, 1), 0, 1) * 255)
        preview(output / (stem + ".png"), source, displayed_pan, target, base_h, reference_h)
        records.append(row)
        print(json.dumps(row), flush=True)
    (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

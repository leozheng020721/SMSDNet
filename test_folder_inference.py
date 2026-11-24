"""Offline checks for ENVI discovery, tiled prediction, and export geometry."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import folder_inference


def write_envi(directory, stem="image_1_rectified", height=19, width=31, valid=None):
    directory.mkdir(parents=True, exist_ok=True)
    rows, columns = np.indices((height, width), dtype=np.uint16)
    data = np.stack((
        300 + columns * 7 + rows,
        600 + rows * 12 + columns * 2,
        1200 + rows * 3 + columns * 19,
        50000 + (rows + columns) % 10000,
    ), axis=-1).astype("<u2")
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    data[~valid] = 65535
    data_path = directory / (stem + "_envi.dat")
    header_path = data_path.with_suffix(".hdr")
    data_path.write_bytes(data.tobytes())
    header_path.write_text(
        "ENVI\n"
        f"samples = {width}\nlines = {height}\nbands = 4\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        "data type = 12\ninterleave = bip\nbyte order = 0\n"
        "band names = {BLUE, GREEN, RED, NIR}\ndefault bands = {3, 2, 1}\n",
        encoding="ascii",
    )
    valid_path = directory / (stem + "_valid_mask.dat")
    valid_path.write_bytes((valid.astype(np.uint8) * 255).tobytes())
    valid_path.with_suffix(".hdr").write_text(
        "ENVI\n"
        f"samples = {width}\nlines = {height}\nbands = 1\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        "data type = 1\ninterleave = bip\nbyte order = 0\n",
        encoding="ascii",
    )
    return header_path, data, valid


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


class FolderInferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="folder_inference_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        self.output = self.root / "output"

    def test_envi_bip_selects_rgb_and_reuses_global_valid_pixel_stretch(self):
        valid = np.ones((19, 31), dtype=bool)
        valid[:3, :4] = False
        header, data, valid = write_envi(self.input / "region_01", valid=valid)
        records = folder_inference.discover_sources(self.input)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(Path(record["source_path"]), header.resolve())
        self.assertEqual((record["width"], record["height"], record["bands"]), (31, 19, 4))
        self.assertEqual(record["dtype"], "uint16")
        self.assertEqual(list(record["rgb_indices"]), [2, 1, 0])
        with folder_inference.open_source(record) as reader:
            rgb = reader.read_rgb(0, 0, 31, 19)
            actual_valid = reader.read_valid(0, 0, 31, 19)
            left = reader.read_rgb(0, 0, 13, 19)
            right = reader.read_rgb(13, 0, 18, 19)
            self.assertTrue(reader.stretch_summary)
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(actual_valid, valid)
        np.testing.assert_array_equal(np.concatenate((left, right), axis=1), rgb)
        raw_rgb = data[:, :, [2, 1, 0]].astype(np.float64)
        low, high = np.percentile(raw_rgb[valid], [2, 98], axis=0)
        expected = np.clip((raw_rgb - low) / (high - low), 0, 1) * 255
        np.testing.assert_allclose(rgb[valid], expected[valid], atol=1.0, rtol=0)

    def test_discovery_deduplicates_formats_and_ignores_previews_and_valid_masks(self):
        region = self.input / "region_01"
        header, _, _ = write_envi(region)
        preview = Image.fromarray(np.full((19, 31, 3), 100, dtype=np.uint8))
        preview.save(region / "image_1_rectified.tif")
        preview.save(region / "image_1_rectified.jpg")
        preview.save(region / "overlay_preview.jpg")
        preview.save(region / "pair_preview.jpg")
        preview.save(region / "image_2_rectified.tif")
        second_header, _, _ = write_envi(self.input / "region_02")
        records = folder_inference.discover_sources(self.input)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            {Path(record["source_path"]) for record in records},
            {header.resolve(), second_header.resolve(), (region / "image_2_rectified.tif").resolve()},
        )
        self.assertEqual(len({(record["relative_parent"], record["stem"]) for record in records}), 3)

    def test_uint8_tiff_fallback_keeps_rgb_values_and_has_valid_pixels(self):
        self.input.mkdir()
        image = np.zeros((11, 23, 3), dtype=np.uint8)
        image[:, :, 0] = 221
        image[:, :, 1] = 87
        image[:, :, 2] = 16
        Image.fromarray(image).save(self.input / "image_1_rectified.tif")
        records = folder_inference.discover_sources(self.input)
        self.assertEqual(len(records), 1)
        with folder_inference.open_source(records[0]) as reader:
            np.testing.assert_array_equal(reader.read_rgb(0, 0, 23, 11), image)
            np.testing.assert_array_equal(reader.read_valid(0, 0, 23, 11), np.ones((11, 23), dtype=bool))

    def test_nonoverlapping_manifest_covers_each_original_pixel_once(self):
        for length in (1, 511, 512, 513, 1024, 1025):
            write_envi(self.input / f"case_{length}", height=3, width=length)
        result = folder_inference.scan_folder(self.input, self.output, tile_size=512)
        rows = read_manifest(result["tile_manifest"])
        for length in (1, 511, 512, 513, 1024, 1025):
            with self.subTest(length=length):
                tiles = [row for row in rows if int(row["source_width"]) == length]
                self.assertEqual(len(tiles), (length + 511) // 512)
                self.assertEqual([int(row["x"]) for row in tiles], list(range(0, length, 512)))
                coverage = np.zeros((3, length), dtype=np.uint8)
                for row in tiles:
                    x, y = int(row["x"]), int(row["y"])
                    width, height = int(row["valid_width"]), int(row["valid_height"])
                    self.assertEqual((int(row["width"]), int(row["height"])), (512, 512))
                    coverage[y:y + height, x:x + width] += 1
                np.testing.assert_array_equal(coverage, np.ones((3, length), dtype=np.uint8))

    def test_inference_windows_anchor_to_far_edge_and_cover_small_images(self):
        for length, stride, expected in (
            (100, 384, [0]), (512, 384, [0]), (513, 512, [0, 1]),
            (1025, 512, [0, 512, 513]), (1200, 384, [0, 384, 688]),
        ):
            with self.subTest(length=length, stride=stride):
                self.assertEqual(list(folder_inference.tile_starts(length, 512, stride)), expected)

    def test_scan_counts_all_sources_but_limits_selected_tiles_without_importing_torch(self):
        write_envi(self.input / "region_01", height=513, width=541)
        write_envi(
            self.input / "region_02", height=17, width=23,
            valid=np.zeros((17, 23), dtype=bool),
        )
        code = (
            "import json,sys; import folder_inference; "
            "assert 'torch' not in sys.modules; "
            "result=folder_inference.scan_folder(sys.argv[1],sys.argv[2],tile_size=512,count=1); "
            "assert 'torch' not in sys.modules; print(json.dumps(result['summary']))"
        )
        environment = dict(os.environ, PYTHONUTF8="1")
        completed = subprocess.run(
            [sys.executable, "-c", code, str(self.input), str(self.output)],
            cwd=Path(__file__).resolve().parent, env=environment,
            capture_output=True, text=True, check=True,
        )
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["source_count"], 2)
        self.assertEqual(summary["total_tiles"], 5)
        self.assertEqual(summary["valid_tiles"], 4)
        self.assertEqual(summary["empty_tiles"], 1)
        self.assertEqual(summary["selected_source_count"], 1)
        self.assertEqual(summary["selected_total_tiles"], 4)
        result = folder_inference.scan_folder(self.input, self.output, tile_size=512, count=1)
        self.assertEqual(len(read_manifest(result["source_manifest"])), 2)
        tiles = read_manifest(result["tile_manifest"])
        self.assertEqual(len(tiles), 4)
        self.assertEqual({(int(row["x"]), int(row["y"])) for row in tiles}, {(0, 0), (512, 0), (0, 512), (512, 512)})
        self.assertEqual(sorted(int(row["valid_width"]) for row in tiles), [29, 29, 512, 512])
        self.assertEqual(sorted(int(row["valid_height"]) for row in tiles), [1, 1, 512, 512])

    def assert_fake_model_export(self, stride):
        import torch

        class PixelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []

            def forward(self, images):
                self.batches.append(tuple(images.shape))
                red = images[:, :1] * 0.229 + 0.485
                return torch.cat((0.5 - red, red - 0.5), dim=1)

        height, width = 529, 541
        valid = np.ones((height, width), dtype=bool)
        valid[30:71, 41:79] = False
        write_envi(self.input / "region_01", height=height, width=width, valid=valid)
        record = folder_inference.discover_sources(self.input)[0]
        with folder_inference.open_source(record) as reader:
            expected_rgb = reader.read_rgb(0, 0, width, height)
        expected_rgb[~valid] = 0
        expected_shadow = ((expected_rgb[:, :, 0] >= 128) & valid).astype(np.uint8) * 255
        config = {
            "dataset": {
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "in_channels": 3, "num_classes": 2, "spatial_mode": "crop",
                "input_size": 512, "tile_stride": stride, "tile_batch_size": 2,
            },
        }
        arguments = SimpleNamespace(
            config="unused.yaml", checkpoint="unused.pth", input=str(self.input),
            output=str(self.output), count=1, tile_size=512, tile_batch_size=2,
            stride=stride, scan_only=False, device="cpu",
        )
        model = PixelModel()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with patch.object(folder_inference, "load_model", return_value=(model, config, torch.device("cpu"))) as load_model:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = folder_inference.run_inference(arguments)
            load_model.assert_called_once()
        finally:
            torch.set_num_threads(previous_threads)
        tiles = read_manifest(result["tile_manifest"])
        self.assertEqual(len(tiles), 4)
        self.assertTrue(model.batches)
        self.assertTrue(all(shape[0] <= 2 and shape[-2:] == (512, 512) for shape in model.batches))
        reconstructed_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        reconstructed_shadow = np.zeros((height, width), dtype=np.uint8)
        coverage = np.zeros((height, width), dtype=np.uint8)
        for row in tiles:
            x, y = int(row["x"]), int(row["y"])
            tile_width, tile_height = int(row["valid_width"]), int(row["valid_height"])
            self.assertEqual((int(row["width"]), int(row["height"])), (512, 512))
            with Image.open(row["rgb_path"]) as image:
                self.assertEqual(image.mode, "RGB")
                rgb = np.asarray(image).copy()
            with Image.open(row["shadow_path"]) as image:
                shadow = np.asarray(image).copy()
            with Image.open(row["overlay_path"]) as image:
                self.assertEqual(image.mode, "RGB")
                overlay = np.asarray(image).copy()
            self.assertEqual(rgb.shape, (512, 512, 3))
            self.assertEqual(shadow.shape, (512, 512))
            self.assertEqual(overlay.shape, (512, 512, 3))
            tile_valid = np.zeros((512, 512), dtype=bool)
            tile_valid[:tile_height, :tile_width] = valid[y:y + tile_height, x:x + tile_width]
            for pixels in (rgb, shadow, overlay):
                self.assertFalse(np.any(pixels[~tile_valid]))
            np.testing.assert_array_equal(
                shadow[:tile_height, :tile_width],
                expected_shadow[y:y + tile_height, x:x + tile_width],
            )
            reconstructed_rgb[y:y + tile_height, x:x + tile_width] = rgb[:tile_height, :tile_width]
            reconstructed_shadow[y:y + tile_height, x:x + tile_width] = shadow[:tile_height, :tile_width]
            coverage[y:y + tile_height, x:x + tile_width] += 1
        np.testing.assert_array_equal(coverage, np.ones((height, width), dtype=np.uint8))
        np.testing.assert_array_equal(reconstructed_rgb, expected_rgb)
        np.testing.assert_array_equal(reconstructed_shadow, expected_shadow)

    def test_cpu_pixel_model_exports_512_tiles_with_black_padding_and_rebuildable_coordinates(self):
        self.assert_fake_model_export(stride=512)

    def test_overlapping_cpu_pixel_predictions_match_dense_pixel_reference(self):
        self.assert_fake_model_export(stride=384)

    def test_context_dependent_overlap_accumulates_logits_and_coverage(self):
        import torch

        class ContextModel(torch.nn.Module):
            def forward(self, images):
                red = images[:, :1] * 0.229 + 0.485
                score = red - red.mean(dim=(-2, -1), keepdim=True) + 0.013
                return torch.cat((-score, score), dim=1)

        self.input.mkdir()
        rows, columns = np.indices((9, 11))
        rgb = np.stack((30 + columns * 18 + rows * 4, np.full_like(rows, 100), np.full_like(rows, 40)), axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(self.input / "image_1_rectified.tif")
        sums = np.zeros((2, 9, 11), dtype=np.float64)
        counts = np.zeros((9, 11), dtype=np.uint32)
        overwritten = np.zeros_like(sums)
        # Enumerate the four spatial windows independently of the tiling helper.
        for x, y in ((0, 0), (3, 0), (0, 1), (3, 1)):
            red = rgb[y:y + 8, x:x + 8, 0].astype(np.float64) / 255.0
            score = red - red.mean() + 0.013
            logits = np.stack((-score, score), axis=0)
            sums[:, y:y + 8, x:x + 8] += logits
            counts[y:y + 8, x:x + 8] += 1
            overwritten[:, y:y + 8, x:x + 8] = logits
        averaged = sums / counts[None]
        expected = averaged.argmax(axis=0).astype(np.uint8) * 255
        last_window_only = overwritten.argmax(axis=0).astype(np.uint8) * 255
        self.assertTrue(np.any(expected != last_window_only))
        captured = {}
        close_map = folder_inference._close_map

        def capture_and_close(array):
            if array is not None and array.shape == (9, 11):
                captured["coverage"] = np.array(array, copy=True)
            elif array is not None and array.shape == (2, 9, 11):
                captured["sums"] = np.array(array, copy=True)
            close_map(array)

        config = {"dataset": {"num_classes": 2, "normalization": {
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        }}}
        args = SimpleNamespace(
            config="unused.yaml", checkpoint="unused.pth", input=str(self.input),
            output=str(self.output), count=None, tile_size=8, tile_batch_size=3,
            stride=5, scan_only=False, device="cpu",
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with patch.object(folder_inference, "load_model", return_value=(ContextModel(), config, torch.device("cpu"))), \
                    patch.object(folder_inference, "_close_map", side_effect=capture_and_close), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = folder_inference.run_inference(args)
        finally:
            torch.set_num_threads(previous_threads)
        np.testing.assert_array_equal(captured["coverage"], counts)
        np.testing.assert_allclose(captured["sums"], sums, atol=1e-6, rtol=0)
        rebuilt = np.zeros((9, 11), dtype=np.uint8)
        for row in read_manifest(result["tile_manifest"]):
            x, y = int(row["x"]), int(row["y"])
            width, height = int(row["valid_width"]), int(row["valid_height"])
            with Image.open(row["shadow_path"]) as image:
                tile = np.asarray(image)
                rebuilt[y:y + height, x:x + width] = tile[:height, :width]
        np.testing.assert_array_equal(rebuilt, expected)

    def test_paired_export_matches_filenames_and_outlines_without_filling_interiors(self):
        import torch

        class RectangleModel(torch.nn.Module):
            def forward(self, images):
                score = images.new_full((images.shape[0], 1, *images.shape[-2:]), -1.0)
                score[:, :, 10:96, 12:96] = 1.0
                return torch.cat((-score, score), dim=1)

        expected_by_stem = {}
        for image_number, (height, width) in enumerate(((40, 52), (37, 47)), start=1):
            stem = f"image_{image_number}_rectified"
            valid = np.ones((height, width), dtype=bool)
            valid[:4, :6] = False
            valid[18:22, 20:24] = False
            write_envi(self.input, stem=stem, height=height, width=width, valid=valid)
            expected_by_stem[stem] = {"valid": valid, "height": height, "width": width}
        for record in folder_inference.discover_sources(self.input):
            with folder_inference.open_source(record) as reader:
                expected_by_stem[record["stem"]]["rgb"] = reader.read_rgb(
                    0, 0, reader.width, reader.height,
                )
        config = {"dataset": {"num_classes": 2, "normalization": {
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        }}}
        args = SimpleNamespace(
            config="unused.yaml", checkpoint="unused.pth", input=str(self.input),
            output=str(self.output), count=None, tile_size=512, tile_batch_size=2,
            stride=384, scan_only=False, device="cpu", export_layout="paired",
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with patch.object(folder_inference, "load_model", return_value=(RectangleModel(), config, torch.device("cpu"))), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = folder_inference.run_inference(args)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual({path.name for path in self.output.iterdir() if path.is_dir()}, {"masks", "marked"})
        masks = {path.name for path in (self.output / "masks").iterdir()}
        marked = {path.name for path in (self.output / "marked").iterdir()}
        self.assertEqual(masks, marked)
        self.assertEqual(masks, {
            "image_1_rectified_r00000_c00000.png", "image_2_rectified_r00000_c00000.png",
        })
        self.assertEqual(len(list(self.output.rglob("*.png"))), 4)
        records = read_manifest(result["tile_manifest"])
        self.assertEqual(len(records), 2)
        self.assertEqual(len({row["tile_id"] for row in records}), 2)
        for row in records:
            expected = expected_by_stem[row["stem"]]
            height, width = expected["height"], expected["width"]
            valid = expected["valid"]
            self.assertEqual((int(row["x"]), int(row["y"])), (0, 0))
            self.assertEqual((int(row["valid_width"]), int(row["valid_height"])), (width, height))
            self.assertEqual(row["rgb_path"], "")
            self.assertEqual(row["overlay_path"], "")
            mask_path, marked_path = Path(row["shadow_path"]), Path(row["marked_path"])
            self.assertEqual(mask_path.parent, self.output.resolve() / "masks")
            self.assertEqual(marked_path.parent, self.output.resolve() / "marked")
            self.assertEqual(mask_path.name, marked_path.name)
            with Image.open(mask_path) as image:
                self.assertEqual(image.mode, "L")
                mask = np.asarray(image).copy()
            with Image.open(marked_path) as image:
                self.assertEqual(image.mode, "RGB")
                annotated = np.asarray(image).copy()
            self.assertEqual(mask.shape, (512, 512))
            self.assertEqual(annotated.shape, (512, 512, 3))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            tile_valid = np.zeros((512, 512), dtype=bool)
            tile_valid[:height, :width] = valid
            foreground = np.zeros((512, 512), dtype=bool)
            foreground[10:96, 12:96] = True
            foreground &= tile_valid
            np.testing.assert_array_equal(mask, foreground.astype(np.uint8) * 255)
            base_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            base_rgb[:height, :width] = expected["rgb"]
            self.assertFalse(np.any(mask[~tile_valid]))
            self.assertFalse(np.any(annotated[~tile_valid]))
            changed = np.any(annotated != base_rgb, axis=-1)
            self.assertTrue(changed.any())
            np.testing.assert_array_equal(annotated[changed], np.tile([255, 0, 0], (int(changed.sum()), 1)))
            np.testing.assert_array_equal(annotated[10, 12], [255, 0, 0])
            # The center stays in the original colors; only the boundary is red.
            np.testing.assert_array_equal(annotated[26:30, 30:34], base_rgb[26:30, 30:34])
            self.assertTrue(np.all(mask[26:30, 30:34] == 255))
            np.testing.assert_array_equal(annotated[6, 8], base_rgb[6, 8])

    def test_paired_scan_only_plans_matching_paths_without_torch_or_png_exports(self):
        write_envi(self.input, stem="image_1_rectified")
        write_envi(self.input, stem="image_2_rectified")
        code = (
            "import json,sys; from types import SimpleNamespace; import folder_inference; "
            "assert 'torch' not in sys.modules; "
            "args=SimpleNamespace(input=sys.argv[1],output=sys.argv[2],tile_size=512,"
            "count=None,scan_only=True,export_layout='paired'); "
            "result=folder_inference.run_inference(args); "
            "assert 'torch' not in sys.modules; print(json.dumps(result['summary']))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(self.input), str(self.output)],
            cwd=Path(__file__).resolve().parent, env=dict(os.environ, PYTHONUTF8="1"),
            capture_output=True, text=True, check=True,
        )
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["source_count"], 2)
        self.assertEqual(summary["selected_total_tiles"], 2)
        self.assertFalse(list(self.output.rglob("*.png")))
        records = read_manifest(self.output / "tile_manifest.csv")
        self.assertEqual(len(records), 2)
        self.assertEqual(len({row["tile_id"] for row in records}), 2)
        for row in records:
            mask_path, marked_path = Path(row["shadow_path"]), Path(row["marked_path"])
            self.assertEqual(mask_path.name, marked_path.name)
            self.assertEqual(mask_path.parent, self.output.resolve() / "masks")
            self.assertEqual(marked_path.parent, self.output.resolve() / "marked")
            self.assertEqual(row["rgb_path"], "")
            self.assertEqual(row["overlay_path"], "")


if __name__ == "__main__":
    unittest.main()

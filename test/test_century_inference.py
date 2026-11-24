"""CPU integration tests for windowed, georeferenced RGB + NIR inference."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "century_inference_under_test", Path(__file__).with_name("infer_century_nir.py")
)
inference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inference
SPEC.loader.exec_module(inference)


class WindowOnlySource:
    """Fail if inference tries to read a complete raster without a window."""

    def __init__(self, source):
        self.source = source
        self.read_windows = []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _check(self, kwargs):
        window = kwargs.get("window")
        if window is None:
            raise AssertionError("Raster reads must use explicit windows")
        self.read_windows.append(window)

    def read(self, *args, **kwargs):
        self._check(kwargs)
        return self.source.read(*args, **kwargs)

    def read_masks(self, *args, **kwargs):
        self._check(kwargs)
        return self.source.read_masks(*args, **kwargs)

    def dataset_mask(self, *args, **kwargs):
        self._check(kwargs)
        return self.source.dataset_mask(*args, **kwargs)


class NirAwareModel(torch.nn.Module):
    def __init__(self, contextual=False):
        super().__init__()
        self.contextual = contextual
        self.calls = 0
        self.nonzero_nir_seen = False
        self.invalid_seen = False

    def forward(self, rgb, nir=None, nir_valid=None):
        if nir is None or nir_valid is None:
            raise AssertionError("The NIR branch must receive both inputs")
        if nir_valid.dtype != torch.bool:
            raise AssertionError("NIR validity must retain boolean semantics")
        if rgb.shape[1] != 3 or nir.shape[1] != 1:
            raise AssertionError("Incorrect RGB/NIR channel layout")
        if torch.any(nir[~nir_valid] != 0):
            raise AssertionError("Invalid NIR pixels must be zero")
        self.calls += 1
        self.nonzero_nir_seen |= bool(torch.any(nir[nir_valid] != 0))
        self.invalid_seen |= bool(torch.any(~nir_valid))
        score = nir * 0.9 + rgb[:, :1] * 0.6 - 0.15
        if self.contextual:
            context = rgb[:, :1].mean(dim=(-2, -1), keepdim=True)
            score = score + 7.0 * context - 0.5
        return torch.cat((-score, score), dim=1)


class RecordingNirAwareModel(NirAwareModel):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, rgb, nir=None, nir_valid=None):
        self.inputs.append((rgb.clone(), nir.clone(), nir_valid.clone()))
        return super().forward(rgb, nir=nir, nir_valid=nir_valid)


class CenturyInferenceTests(unittest.TestCase):
    MEAN = (0.1, 0.2, 0.3)
    STD = (0.5, 0.25, 0.2)
    DN_MAX = 4095.0

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def make_source(self, height=11, width=13, constant_nir=False):
        rows, cols = np.indices((height, width))
        blue = (200 + rows * 11 + cols * 7).astype(np.uint16)
        green = (600 + rows * 17 + cols * 13).astype(np.uint16)
        red = (1000 + rows * 23 + cols * 19).astype(np.uint16)
        nir = (110 + rows * 137 + cols * 37).astype(np.uint16)
        if constant_nir:
            nir.fill(1234)
        bands = np.stack((blue, green, red, nir))
        valid = np.ones((height, width), dtype=bool)
        valid[0, 0] = False
        valid[height // 2, width // 2] = False
        bands[:, ~valid] = 0
        path = self.root / "source.tif"
        profile = {
            "driver": "GTiff", "width": width, "height": height,
            "count": 4, "dtype": "uint16", "nodata": 0,
            "crs": "EPSG:32650", "transform": from_origin(456123.5, 3456789.5, 0.5, 0.5),
        }
        with rasterio.open(path, "w", **profile) as writer:
            writer.write(bands)
            for index, description in enumerate(("BLUE", "GREEN", "RED", "NIR"), 1):
                writer.set_band_description(index, description)
        return path, bands, valid

    def test_tile_starts_cover_small_and_awkward_dimensions(self):
        self.assertEqual(inference.tile_starts(3, 8, 6), [0])
        self.assertEqual(inference.tile_starts(13, 6, 4), [0, 4, 7])
        self.assertEqual(inference.tile_starts(11, 6, 4), [0, 4, 5])
        for length in (1, 6, 7, 11, 13, 25):
            coverage = np.zeros(length, dtype=np.uint8)
            for left in inference.tile_starts(length, 6, 4):
                coverage[left:min(left + 6, length)] += 1
            self.assertTrue(np.all(coverage > 0))

    def test_histogram_percentiles_match_numpy_linear_interpolation(self):
        values = np.array([1, 1, 3, 7, 9, 9, 12, 65535], dtype=np.uint16)
        histogram = np.bincount(values, minlength=65536)
        percentiles = (0, 1, 2, 13.5, 50, 77.7, 98, 99, 100)
        actual = inference.percentile_from_histogram(histogram, percentiles)
        np.testing.assert_allclose(actual, np.percentile(values, percentiles), rtol=0, atol=1e-10)

    def test_scene_statistics_use_valid_nir_and_windowed_reads(self):
        path, bands, valid = self.make_source()
        with rasterio.open(path) as source:
            wrapped = WindowOnlySource(source)
            actual = inference.scene_statistics(wrapped, (2, 98))
            self.assertTrue(wrapped.read_windows)
        self.assertEqual(actual["valid_pixels"], int(valid.sum()))
        expected = np.percentile(bands[3, valid], (2, 98))
        np.testing.assert_allclose([actual["nir_low"], actual["nir_high"]], expected)

    def test_prepare_tile_rgb_order_normalization_nir_and_padding(self):
        path, bands, valid = self.make_source(height=7, width=9)
        nir_low, nir_high = np.percentile(bands[3, valid], (2, 98))
        with rasterio.open(path) as source:
            rgb, nir, nir_valid, scene_valid = inference.prepare_tile(
                WindowOnlySource(source), 7, 5, 4, self.MEAN, self.STD,
                self.DN_MAX, nir_low, nir_high,
            )
        self.assertEqual(rgb.shape, (3, 4, 4))
        self.assertEqual(nir.shape, (1, 4, 4))
        self.assertEqual(nir_valid.shape, (1, 4, 4))
        self.assertEqual(scene_valid.shape, (4, 4))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertEqual(nir.dtype, np.float32)
        self.assertEqual(nir_valid.dtype, np.bool_)
        self.assertEqual(scene_valid.dtype, np.bool_)
        expected_rgb = bands[[2, 1, 0], 5:7, 7:9].astype(np.float32) / self.DN_MAX
        expected_rgb = (expected_rgb - np.array(self.MEAN, dtype=np.float32)[:, None, None])
        expected_rgb /= np.array(self.STD, dtype=np.float32)[:, None, None]
        np.testing.assert_allclose(rgb[:, :2, :2], expected_rgb, atol=1e-6)
        expected_nir = 2 * np.clip((bands[3, 5:7, 7:9].astype(np.float32) - nir_low)
                                   / (nir_high - nir_low), 0, 1) - 1
        np.testing.assert_allclose(nir[0, :2, :2], expected_nir, atol=1e-6)
        self.assertTrue(np.all(scene_valid[:2, :2]))
        self.assertFalse(np.any(scene_valid[2:, :]))
        self.assertFalse(np.any(scene_valid[:, 2:]))
        self.assertTrue(np.all(nir[~nir_valid] == 0))

    def test_constant_nir_remains_finite_with_validity_preserved(self):
        path, _, valid = self.make_source(height=5, width=5, constant_nir=True)
        with rasterio.open(path) as source:
            statistics = inference.scene_statistics(WindowOnlySource(source), (2, 98))
            _, nir, nir_valid, scene_valid = inference.prepare_tile(
                WindowOnlySource(source), 0, 0, 8, self.MEAN, self.STD,
                self.DN_MAX, statistics["nir_low"], statistics["nir_high"],
            )
        self.assertEqual(statistics["valid_pixels"], int(valid.sum()))
        self.assertTrue(np.all(np.isfinite(nir)))
        np.testing.assert_array_equal(nir_valid[0, :5, :5], valid)
        np.testing.assert_array_equal(scene_valid[:5, :5], valid)
        self.assertTrue(np.all(nir[~nir_valid] == 0))

    def test_zero_nir_with_nonzero_rgb_is_a_valid_spectral_measurement(self):
        path, bands, valid = self.make_source(height=5, width=5)
        bands[3, 1, 1] = 0
        with rasterio.open(path, "r+") as writer:
            writer.write(bands)
        with rasterio.open(path) as source:
            statistics = inference.scene_statistics(WindowOnlySource(source), (2, 98))
            _, nir, nir_valid, scene_valid = inference.prepare_tile(
                WindowOnlySource(source), 0, 0, 8, self.MEAN, self.STD,
                self.DN_MAX, statistics["nir_low"], statistics["nir_high"],
            )
        self.assertEqual(statistics["valid_pixels"], int(valid.sum()))
        self.assertTrue(nir_valid[0, 1, 1])
        self.assertTrue(scene_valid[1, 1])
        self.assertEqual(nir[0, 1, 1], -1)

    def test_all_invalid_scene_has_no_spectral_support(self):
        path, bands, _ = self.make_source(height=5, width=5)
        with rasterio.open(path, "r+") as writer:
            writer.write(np.zeros_like(bands))
        with rasterio.open(path) as source:
            statistics = inference.scene_statistics(WindowOnlySource(source), (2, 98))
            rgb, nir, nir_valid, scene_valid = inference.prepare_tile(
                WindowOnlySource(source), 0, 0, 8, self.MEAN, self.STD,
                self.DN_MAX, statistics["nir_low"], statistics["nir_high"],
            )
        self.assertEqual(statistics["valid_pixels"], 0)
        self.assertEqual((statistics["nir_low"], statistics["nir_high"]), (0, 0))
        self.assertFalse(np.any(nir_valid))
        self.assertFalse(np.any(scene_valid))
        self.assertTrue(np.all(rgb == 0))
        self.assertTrue(np.all(nir == 0))

    def run_stream_case(self, height, width, tile_size, stride, contextual=False):
        path, _, valid = self.make_source(height=height, width=width)
        mask_path = self.root / "shadow.tif"
        valid_path = self.root / "valid.tif"
        probability_path = self.root / "probability.tif"
        with rasterio.open(path) as source:
            statistics = inference.scene_statistics(WindowOnlySource(source), (2, 98))
            profile = inference.output_profile(source)
            probability_profile = inference.output_profile(source, probability=True)
            model = NirAwareModel(contextual=contextual).eval()
            wrapped = WindowOnlySource(source)
            with rasterio.open(mask_path, "w", **profile) as mask_writer, \
                    rasterio.open(valid_path, "w", **profile) as valid_writer, \
                    rasterio.open(probability_path, "w", **probability_profile) as probability_writer:
                actual = inference.stream_predict(
                    model, wrapped, torch.device("cpu"), self.MEAN, self.STD,
                    statistics["nir_low"], statistics["nir_high"],
                    mask_writer, valid_writer, probability_writer=probability_writer,
                    tile_size=tile_size, stride=stride, batch_size=2,
                    dn_max=self.DN_MAX, amp_enabled=False, threshold=0.5,
                )
            self.assertTrue(model.nonzero_nir_seen)
            self.assertTrue(model.invalid_seen)
            self.assertTrue(wrapped.read_windows)
            self.assertTrue(all(window.height <= tile_size and window.width <= tile_size
                                for window in wrapped.read_windows))
            reference_model = NirAwareModel(contextual=contextual).eval()
            accumulated = np.zeros((2, height, width), dtype=np.float32)
            probability_sum = np.zeros((height, width), dtype=np.float32)
            coverage = np.zeros((height, width), dtype=np.float32)
            ys = inference.tile_starts(height, tile_size, stride)
            xs = inference.tile_starts(width, tile_size, stride)
            for y in ys:
                for x in xs:
                    rgb, nir, nir_valid, tile_valid = inference.prepare_tile(
                        source, x, y, tile_size, self.MEAN, self.STD, self.DN_MAX,
                        statistics["nir_low"], statistics["nir_high"],
                    )
                    with torch.no_grad():
                        logits = reference_model(
                            torch.from_numpy(rgb)[None], nir=torch.from_numpy(nir)[None],
                            nir_valid=torch.from_numpy(nir_valid)[None],
                        )[0]
                    crop_height, crop_width = min(tile_size, height - y), min(tile_size, width - x)
                    tile_valid = tile_valid[:crop_height, :crop_width]
                    region = np.s_[y:y + crop_height, x:x + crop_width]
                    tile_logits = logits[:, :crop_height, :crop_width].numpy()
                    accumulated[:, region[0], region[1]] += tile_logits * tile_valid
                    probability_sum[region] += torch.softmax(logits, dim=0)[1, :crop_height, :crop_width].numpy() * tile_valid
                    coverage[region] += tile_valid
            mean_logits = accumulated / np.maximum(coverage, 1)[None]
            expected_probability = torch.softmax(torch.from_numpy(mean_logits), dim=0)[1].numpy()
            expected_probability[~valid] = -1
            expected_mask = ((expected_probability > 0.5) & valid).astype(np.uint8) * 255
            if contextual:
                mean_probability = probability_sum / np.maximum(coverage, 1)
                self.assertGreater(float(np.max(np.abs(mean_probability[valid] - expected_probability[valid]))), 1e-5)
            for output in (mask_path, valid_path, probability_path):
                with rasterio.open(output) as result:
                    self.assertEqual(result.transform, source.transform)
                    self.assertEqual(result.crs, source.crs)
                    self.assertEqual((result.height, result.width), (height, width))
                    self.assertEqual(result.count, 1)
                    self.assertEqual(result.nodata, -1 if output == probability_path else None)
            with rasterio.open(mask_path) as output:
                np.testing.assert_array_equal(output.read(1), expected_mask)
            with rasterio.open(valid_path) as output:
                np.testing.assert_array_equal(output.read(1), valid.astype(np.uint8) * 255)
            with rasterio.open(probability_path) as output:
                np.testing.assert_allclose(output.read(1), expected_probability, rtol=1e-5, atol=2e-6)
            self.assertEqual(actual["tile_count"], len(ys) * len(xs))
            self.assertEqual(actual["valid_pixels"], int(valid.sum()))
            self.assertEqual(actual["shadow_pixels"], int(np.count_nonzero(expected_mask)))

    def test_stream_matches_nir_aware_reference_with_awkward_overlap(self):
        self.run_stream_case(11, 13, 6, 4)

    def test_stream_preserves_grid_for_scene_smaller_than_tile(self):
        self.run_stream_case(3, 4, 8, 6)

    def test_stream_averages_logits_before_softmax(self):
        self.run_stream_case(11, 13, 6, 4, contextual=True)

    def test_disable_nir_preserves_rgb_and_scene_support(self):
        path, _, expected_valid = self.make_source()
        runs = {}
        with rasterio.open(path) as source:
            statistics = inference.scene_statistics(source, (2, 98))
            for disabled in (False, True):
                mode = "disabled" if disabled else "real"
                paths = {name: self.root / f"{mode}_{name}.tif"
                         for name in ("mask", "valid", "probability")}
                model = RecordingNirAwareModel().eval()
                with rasterio.open(paths["mask"], "w", **inference.output_profile(source)) as mask_writer, \
                        rasterio.open(paths["valid"], "w", **inference.output_profile(source)) as valid_writer, \
                        rasterio.open(paths["probability"], "w", **inference.output_profile(source, probability=True)) as probability_writer:
                    result = inference.stream_predict(
                        model, source, torch.device("cpu"), self.MEAN, self.STD,
                        statistics["nir_low"], statistics["nir_high"],
                        mask_writer, valid_writer, probability_writer,
                        tile_size=6, stride=4, batch_size=2, dn_max=self.DN_MAX,
                        amp_enabled=False, disable_nir=disabled,
                    )
                with rasterio.open(paths["valid"]) as valid_output, \
                        rasterio.open(paths["probability"]) as probability_output:
                    support = valid_output.read(1)
                    probability = probability_output.read(1)
                np.testing.assert_array_equal(support, expected_valid.astype(np.uint8) * 255)
                self.assertEqual(result["valid_pixels"], int(expected_valid.sum()))
                self.assertTrue(np.all(np.isfinite(probability[expected_valid])))
                self.assertTrue(np.all(probability[~expected_valid] == -1))
                runs[disabled] = (model, probability)
        real, real_probability = runs[False]
        disabled, disabled_probability = runs[True]
        self.assertTrue(real.nonzero_nir_seen)
        self.assertEqual(len(real.inputs), len(disabled.inputs))
        for real_inputs, disabled_inputs in zip(real.inputs, disabled.inputs):
            torch.testing.assert_close(real_inputs[0], disabled_inputs[0], rtol=0, atol=0)
            self.assertEqual(disabled_inputs[2].dtype, torch.bool)
            self.assertFalse(bool(torch.any(disabled_inputs[2])))
            self.assertFalse(bool(torch.any(disabled_inputs[1] != 0)))
        self.assertGreater(float(np.max(np.abs(
            real_probability[expected_valid] - disabled_probability[expected_valid]))), 1e-3)

    def test_discover_inputs_filters_exact_group_and_region(self):
        for group in (1, 4):
            for region in ("region_01", "region_010", "region_02"):
                folder = self.root / str(group) / region
                folder.mkdir(parents=True)
                for name in ("image_1_rectified.tif", "image_2_rectified.tif",
                             "image_3_rectified.tif", "image_1_rectified_envi.dat"):
                    (folder / name).touch()
        expected = [self.root / "4" / "region_01" / f"image_{index}_rectified.tif"
                    for index in (1, 2)]
        self.assertEqual(inference.discover_inputs(self.root, [4], ["region_01"]), expected)
        self.assertEqual(inference.discover_inputs(self.root, [4], ["region_01", "region_01"]), expected)
        self.assertEqual(len(inference.discover_inputs(self.root, [1, 4])), 12)

    def test_discover_inputs_rejects_missing_or_nonexact_region(self):
        for region in ("region_01", "region_02"):
            folder = self.root / "4" / region
            folder.mkdir(parents=True)
            for index in (1, 2):
                (folder / f"image_{index}_rectified.tif").touch()
        with self.assertRaises(FileNotFoundError):
            inference.discover_inputs(self.root, [4], ["region_01", "region_03"])
        for region in ("region_*", "other", "../region_01"):
            with self.subTest(region=region), self.assertRaises((ValueError, FileNotFoundError)):
                inference.discover_inputs(self.root, [4], [region])

    def test_legacy_signature_only_matches_real_nir_without_mutation(self):
        legacy = {"tile_size": 512, "source": "scene.tif"}
        real = dict(legacy, nir_mode="real")
        disabled = dict(legacy, nir_mode="disabled")
        self.assertTrue(inference.signatures_match(legacy, real))
        self.assertNotIn("nir_mode", legacy)
        self.assertTrue(inference.signatures_match(real, real))
        self.assertTrue(inference.signatures_match(disabled, disabled))
        self.assertFalse(inference.signatures_match(legacy, disabled))
        self.assertFalse(inference.signatures_match(real, disabled))
        self.assertFalse(inference.signatures_match(disabled, real))
        self.assertFalse(inference.signatures_match(None, real))


if __name__ == "__main__":
    unittest.main()

"""CPU checks for optional, geometrically synchronized SEO NIR inputs."""

import copy
import contextlib
import csv
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch

from dataset import (SEONIRDataset, get_nir_transforms, get_seo_dataset, get_seo_loaders,
                     get_transforms, load_aligned_nir, resolve_seo_nir_paths)


def config_for(root, mode='crop', size=8):
    return {
        'random_seed': 42,
        'dataset': {
            'type': 'seo', 'root_dir': str(root), 'val_ratio': 0.4,
            'spatial_mode': mode, 'input_size': size,
            'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
            'nir': {'enabled': True, 'root_dir': str(Path(root) / 'nir_aligned_v1'),
                    'manifest': 'registration_with_candidates.csv',
                    'include_candidates': False, 'normalization': {'percentiles': [0, 100]}},
        },
    }


def save_nir(root, values, valid=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nir_path, valid_path = root / 'sample_nir.tif', root / 'sample_valid.png'
    Image.fromarray(np.asarray(values, np.float32)).save(nir_path)
    valid = np.isfinite(values) if valid is None else valid
    Image.fromarray(np.asarray(valid, np.uint8) * 255).save(valid_path)
    return str(nir_path), str(valid_path)


def joint_compose(operations):
    return A.Compose(operations, additional_targets={'nir': 'image', 'nir_valid': 'image'})


def sample_arrays(height=9, width=11):
    rows, columns = np.indices((height, width))
    code = (rows * width + columns).astype(np.uint8)
    image = np.stack((code, code, code), axis=-1)
    return image, (code % 2).astype(np.uint8), code.astype(np.float32) / 255 * 2 - 1, np.ones(code.shape, bool)


class LoadNIRTest(unittest.TestCase):
    def test_pil_finite_checks_are_row_bounded_and_reuse_png_support_buffer(self):
        raw = np.full((273, 7), 700, np.float32)
        raw[0, 1], raw[129, 2], raw[258, 3] = np.nan, np.inf, -np.inf
        png = np.tile(np.uint8([0, 1, 2, 127, 255, 0, 1]), (raw.shape[0], 1))
        original_raw = raw.copy()
        expected_valid = (png > 0) & np.isfinite(raw)
        original_isfinite = np.isfinite
        inspected = []

        def inspect_isfinite(values, *args, **kwargs):
            array = np.asarray(values)
            result = original_isfinite(values, *args, **kwargs)
            if array.ndim == 2:
                self.assertLessEqual(array.shape[0], 128)
                self.assertEqual(array.shape[1], raw.shape[1])
                self.assertEqual(array.dtype, np.float32)
                self.assertEqual(result.dtype, np.bool_)
                inspected.append(array.shape)
            return result

        with mock.patch.dict(sys.modules, {'rasterio': None}), \
                mock.patch.object(Image, 'open', return_value=contextlib.nullcontext(raw)), \
                mock.patch.object(cv2, 'imread', return_value=png), \
                mock.patch.object(np, 'isfinite', new=inspect_isfinite):
            normalized, valid = load_aligned_nir('synthetic_nir.tif', 'synthetic_valid.png', raw.shape, {})
        self.assertTrue(inspected)
        self.assertTrue(np.shares_memory(valid, png))
        self.assertEqual(valid.dtype, np.bool_)
        self.assertTrue(set(np.unique(png)) <= {0, 1})
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(raw, original_raw)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.all(normalized == 0))

    def test_rasterio_float32_read_intersects_internal_png_and_finite_support(self):
        raw = np.random.default_rng(31).uniform(300, 8000, (257, 7)).astype(np.float32)
        raw[0, 1], raw[129, 2], raw[256, 4] = np.nan, np.inf, -np.inf
        original_raw = raw.copy()
        png = np.tile(np.uint8([0, 1, 2, 127, 255, 1, 255]), (raw.shape[0], 1))
        internal = np.full(raw.shape, 255, np.uint8)
        internal[1::3, 1], internal[::4, 4] = 0, 0
        internal[2::3, 2] = 1
        original_internal = internal.copy()
        expected_valid = (png > 0) & (internal > 0) & np.isfinite(raw)
        low, high = np.percentile(raw[expected_valid].astype(np.float64), [0, 100])
        expected = np.zeros(raw.shape, np.float32)
        expected[expected_valid] = 2 * np.clip((raw[expected_valid].astype(np.float64) - low)
                                             / (high - low), 0, 1) - 1
        source = mock.MagicMock()
        source.__enter__.return_value = source
        source.count, source.read.return_value, source.read_masks.return_value = 1, raw, internal
        fake_rasterio, fake_errors = ModuleType('rasterio'), ModuleType('rasterio.errors')
        fake_rasterio.open = mock.Mock(return_value=source)
        fake_errors.NotGeoreferencedWarning = UserWarning
        with mock.patch.dict(sys.modules, {'rasterio': fake_rasterio, 'rasterio.errors': fake_errors}), \
                mock.patch.object(cv2, 'imread', return_value=png):
            normalized, valid = load_aligned_nir('synthetic_nir.tif', 'synthetic_valid.png', raw.shape,
                                                {'normalization': {'percentiles': [0, 100]}})
        source.read.assert_called_once_with(1, out_dtype='float32')
        source.read_masks.assert_called_once_with(1)
        self.assertTrue(np.shares_memory(valid, png))
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(internal, original_internal)
        np.testing.assert_allclose(normalized, expected, rtol=2e-6, atol=2e-6)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.all(normalized[~valid] == 0))

    def test_missing_rasterio_does_not_attach_import_error_to_pil_memory_error(self):
        active_errors = []

        def fail_read(path):
            active_errors.append(sys.exc_info()[1])
            raise MemoryError('Synthetic PIL allocation failure')

        with mock.patch.dict(sys.modules, {'rasterio': None}), \
                mock.patch.object(Image, 'open', side_effect=fail_read):
            with self.assertRaisesRegex(MemoryError, 'Synthetic PIL allocation failure') as raised:
                load_aligned_nir('synthetic_nir.tif', 'synthetic_valid.png', (273, 7), {})
        self.assertEqual(active_errors, [None])
        self.assertIsNone(raised.exception.__context__)

    @staticmethod
    def normalization_example():
        raw = np.random.default_rng(711).uniform(100, 60000, (19, 23)).astype(np.float32)
        png_valid = np.indices(raw.shape).sum(axis=0) % 5 != 0
        raw[2, 3], raw[3, 5] = np.nan, np.inf
        raw[1, 4], png_valid[1, 4] = 1e9, False
        valid = png_valid & np.isfinite(raw)
        percentiles = [7.3, 92.6]
        values = raw[valid].astype(np.float64)
        low, high = np.percentile(values, percentiles)
        reference = np.zeros(raw.shape, np.float32)
        reference[valid] = (2 * np.clip((values - low) / (high - low), 0, 1) - 1).astype(np.float32)
        return raw, png_valid, valid, percentiles, reference

    def test_percentile_partitions_a_private_float32_copy_without_reordering_pixels(self):
        raw, png_valid, expected_valid, percentiles, reference = self.normalization_example()
        original = raw.copy()
        original_percentile = np.percentile
        inspected = []

        def inspect_percentile(values, requested, **kwargs):
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.ndim, 1)
            self.assertEqual(values.size, int(expected_valid.sum()))
            self.assertTrue(values.flags.owndata)
            self.assertIsNone(values.base)
            self.assertFalse(np.shares_memory(values, raw))
            self.assertIs(kwargs.get('overwrite_input'), True)
            np.testing.assert_array_equal(values, raw[expected_valid])
            inspected.append(values.size)
            return original_percentile(values, requested, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            paths = save_nir(directory, raw, png_valid)
            # Borrow the source buffer in the PIL path to expose unsafe partitioning.
            with mock.patch.dict(sys.modules, {'rasterio': None}), \
                    mock.patch.object(Image, 'open', return_value=contextlib.nullcontext(raw)), \
                    mock.patch.object(np, 'percentile', side_effect=inspect_percentile):
                normalized, valid = load_aligned_nir(*paths, raw.shape,
                    {'normalization': {'percentiles': percentiles}})
        self.assertEqual(inspected, [int(expected_valid.sum())])
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_allclose(normalized, reference, rtol=2e-6, atol=2e-6)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.all(normalized[~valid] == 0))

    def test_normalization_ufuncs_use_float32_and_the_same_explicit_output_buffer(self):
        raw, png_valid, expected_valid, percentiles, reference = self.normalization_example()
        originals = {name: getattr(np, name) for name in ('subtract', 'divide', 'clip', 'multiply')}
        original_percentile = np.percentile
        in_percentile = False
        operations = []

        def percentile(values, requested, **kwargs):
            nonlocal in_percentile
            in_percentile = True
            try:
                return original_percentile(values, requested, **kwargs)
            finally:
                in_percentile = False

        def traced(name):
            def operation(*args, **kwargs):
                if not in_percentile:
                    out = kwargs.get('out')
                    self.assertIsInstance(out, np.ndarray, name)
                    self.assertEqual(out.dtype, np.float32, name)
                    self.assertEqual(out.shape, raw.shape, name)
                    for value in args:
                        if isinstance(value, np.ndarray):
                            self.assertEqual(value.dtype, np.float32, name)
                        elif np.isscalar(value):
                            self.assertEqual(np.asarray(value).dtype, np.float32, name)
                    operations.append((name, out))
                return originals[name](*args, **kwargs)
            return operation

        with tempfile.TemporaryDirectory() as directory:
            paths = save_nir(directory, raw, png_valid)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {'rasterio': None}))
                stack.enter_context(mock.patch.object(Image, 'open', return_value=contextlib.nullcontext(raw)))
                stack.enter_context(mock.patch.object(np, 'percentile', side_effect=percentile))
                for name in originals:
                    stack.enter_context(mock.patch.object(np, name, new=traced(name)))
                normalized, valid = load_aligned_nir(*paths, raw.shape,
                    {'normalization': {'percentiles': percentiles}})
        self.assertEqual([name for name, _ in operations], ['subtract', 'divide', 'clip', 'multiply', 'subtract'])
        self.assertTrue(all(out is normalized for _, out in operations))
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_allclose(normalized, reference, rtol=2e-6, atol=2e-6)
        self.assertTrue(np.all(normalized[~valid] == 0))

    def test_normalization_uses_full_valid_image_and_excludes_png_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = np.float32([[10, 20, 30], [np.nan, 50, 10000]])
            valid = np.ones(raw.shape, bool)
            valid[1, 2] = False
            paths = save_nir(directory, raw, valid)
            nir, support = load_aligned_nir(*paths, raw.shape, {'normalization': {'percentiles': [0, 100]}})
            np.testing.assert_array_equal(support, [[True, True, True], [False, True, False]])
            np.testing.assert_allclose(nir, [[-1, -.5, 0], [0, 1, 0]], atol=1e-7)
            self.assertEqual(nir.dtype, np.float32)
            self.assertEqual(support.dtype, np.bool_)

    def test_pil_fallback_without_rasterio_matches_float32_and_nan_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = np.float32([[320, np.nan], [400, 1800]])
            paths = save_nir(directory, raw)
            with mock.patch.dict(sys.modules, {'rasterio': None}):
                nir, valid = load_aligned_nir(*paths, raw.shape, {'normalization': {'percentiles': [0, 100]}})
            self.assertEqual(nir.dtype, np.float32)
            self.assertFalse(valid[0, 1])
            self.assertEqual(float(nir[0, 1]), 0)
            self.assertEqual(float(nir[0, 0]), -1)
            self.assertEqual(float(nir[1, 1]), 1)
            self.assertTrue(np.isfinite(nir).all())

    def test_constant_and_all_invalid_images_remain_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            for raw in (np.full((3, 5), 700, np.float32), np.full((3, 5), np.nan, np.float32)):
                with self.subTest(raw=raw[0, 0]):
                    paths = save_nir(directory, raw)
                    nir, valid = load_aligned_nir(*paths, raw.shape, {})
                    self.assertTrue(np.isfinite(nir).all())
                    self.assertTrue(np.all(nir == 0))
                    self.assertEqual(bool(valid.any()), bool(np.isfinite(raw).any()))

    def test_misaligned_shapes_and_invalid_percentiles_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = save_nir(directory, np.ones((3, 5), np.float32))
            with self.assertRaisesRegex(ValueError, 'spatial sizes differ'):
                load_aligned_nir(*paths, (3, 4), {})
            with self.assertRaisesRegex(ValueError, 'percentiles'):
                load_aligned_nir(*paths, (3, 5), {'normalization': {'percentiles': [99, 1]}})


class SEONIRPrefetchTest(unittest.TestCase):
    @staticmethod
    def loaders(workers, prefetch=None):
        config = config_for('.')
        config.update(num_workers=workers, pin_memory=False, training={'batch_size': 6})
        if prefetch is not None:
            config['prefetch_factor'] = prefetch
        dataset = torch.utils.data.TensorDataset(torch.zeros(12, 3, 2, 2))
        with mock.patch('dataset.get_seo_dataset', return_value=dataset), \
                mock.patch('dataset.DataLoader', wraps=torch.utils.data.DataLoader) as create:
            loaders = get_seo_loaders(config)
        return loaders, create.call_args_list

    def test_configured_prefetch_one_applies_to_both_worker_loaders(self):
        (train, val), calls = self.loaders(workers=2, prefetch=1)
        self.assertEqual((train.prefetch_factor, val.prefetch_factor), (1, 1))
        self.assertEqual((train.num_workers, val.num_workers), (2, 2))
        self.assertEqual((train.batch_size, val.batch_size), (6, 1))
        self.assertTrue(all(call.kwargs['prefetch_factor'] == 1 for call in calls))

    def test_omitted_prefetch_keeps_pytorch_default_two_for_workers(self):
        (train, val), calls = self.loaders(workers=2)
        self.assertEqual((train.prefetch_factor, val.prefetch_factor), (2, 2))
        self.assertTrue(all('prefetch_factor' not in call.kwargs for call in calls))

    def test_zero_workers_does_not_receive_prefetch_keyword(self):
        (train, val), calls = self.loaders(workers=0, prefetch=1)
        self.assertIsNone(train.prefetch_factor)
        self.assertIsNone(val.prefetch_factor)
        self.assertTrue(all('prefetch_factor' not in call.kwargs for call in calls))


class JointNIRTransformTest(unittest.TestCase):
    def test_random_crop_flips_and_rotation_preserve_all_four_coordinates(self):
        config = config_for('.', size=5)
        transform = get_nir_transforms(config, 'train')
        transform.geometry = joint_compose([
            A.RandomCrop(height=5, width=5), A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1), A.RandomRotate90(p=1),
        ])
        transform.rgb = A.Compose([A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()])
        image, mask, nir, valid = sample_arrays()
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        torch.testing.assert_close(result['nir'][0], result['image'][0] * 2 - 1)
        decoded = torch.round(result['image'][0] * 255).long()
        torch.testing.assert_close(result['mask'], decoded % 2)
        self.assertTrue(torch.all(result['nir_valid']))

    def test_padding_is_ignored_label_but_never_valid_nir(self):
        transform = get_nir_transforms(config_for('.', size=8), 'train')
        image, mask, nir, valid = sample_arrays(3, 5)
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        support = result['nir_valid'][0]
        self.assertEqual(int(support.sum()), 15)
        torch.testing.assert_close(support, result['mask'] != 255)
        self.assertTrue(torch.all(result['nir'][:, ~support] == 0))
        self.assertEqual(tuple(result['nir'].shape), (1, 8, 8))

    def test_rgb_brightness_and_normalization_do_not_change_nir(self):
        config = config_for('.')
        first, second = get_nir_transforms(config, 'val'), get_nir_transforms(config, 'val')
        second.rgb = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(.2, .2), contrast_limit=(0, 0), p=1),
            A.Normalize(mean=[.5, .5, .5], std=[.25, .25, .25]), ToTensorV2(),
        ])
        image, mask, nir, valid = sample_arrays()
        before = first(image=image, mask=mask, nir=nir, nir_valid=valid)
        after = second(image=image, mask=mask, nir=nir, nir_valid=valid)
        torch.testing.assert_close(before['nir'], after['nir'])
        torch.testing.assert_close(before['nir_valid'], after['nir_valid'])
        self.assertFalse(torch.equal(before['image'], after['image']))

    def test_resize_conservatively_intersects_interpolated_valid_support(self):
        transform = get_nir_transforms(config_for('.', mode='resize'), 'val')
        image, mask, nir, valid = sample_arrays(3, 3)
        valid[1, 1], nir[1, 1] = False, 0
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        expected_valid = cv2.resize(valid.astype(np.float32), (8, 8), interpolation=cv2.INTER_LINEAR) >= 1 - 1e-6
        expected_nir = cv2.resize(nir, (8, 8), interpolation=cv2.INTER_LINEAR)
        expected_nir[~expected_valid] = 0
        np.testing.assert_array_equal(result['nir_valid'][0].numpy(), expected_valid)
        np.testing.assert_allclose(result['nir'][0].numpy(), expected_nir, atol=1e-7)
        self.assertEqual(tuple(result['image'].shape), (3, 8, 8))
        self.assertEqual(result['mask'].dtype, torch.int64)

    def test_validation_crop_mode_preserves_full_size_and_types(self):
        transform = get_nir_transforms(config_for('.', size=5), 'val')
        image, mask, nir, valid = sample_arrays(13, 17)
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        self.assertEqual(tuple(result['image'].shape), (3, 13, 17))
        self.assertEqual(result['nir'].dtype, torch.float32)
        self.assertEqual(result['nir_valid'].dtype, torch.bool)
        np.testing.assert_array_equal(result['mask'].numpy(), mask)
        np.testing.assert_array_equal(result['nir'][0].numpy(), nir)
        self.assertGreaterEqual(float(result['nir'].min()), -1)
        self.assertLessEqual(float(result['nir'].max()), 1)

    def test_crop_retries_only_insufficient_support_and_stops_when_sufficient(self):
        config = config_for('.')
        config['dataset']['nir'].update(crop_retries=4, min_valid_fraction=.5)
        transform = get_nir_transforms(config, 'train')
        image, mask, nir, valid = sample_arrays(4, 4)
        zero = np.zeros((4, 4), np.float32)
        quarter, half = zero.copy(), zero.copy()
        quarter[:1], half[:2] = 1, 1
        transform.geometry = mock.Mock(side_effect=[
            {'image': image, 'mask': mask, 'nir': nir, 'nir_valid': support}
            for support in (zero, quarter, half)
        ])
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        self.assertEqual(transform.geometry.call_count, 3)
        self.assertEqual(int(result['nir_valid'].sum()), 8)
        transform.geometry = mock.Mock(return_value={
            'image': image, 'mask': mask, 'nir': nir, 'nir_valid': valid.astype(np.float32)})
        transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        self.assertEqual(transform.geometry.call_count, 1)

    def test_all_invalid_source_is_kept_zero_without_wasting_crop_retries(self):
        transform = get_nir_transforms(config_for('.'), 'train')
        image, mask, nir, valid = sample_arrays(4, 4)
        valid[:] = False
        transform.geometry = mock.Mock(return_value={
            'image': image, 'mask': mask, 'nir': nir, 'nir_valid': np.zeros((4, 4), np.float32)})
        result = transform(image=image, mask=mask, nir=nir, nir_valid=valid)
        self.assertEqual(transform.geometry.call_count, 1)
        self.assertFalse(result['nir_valid'].any())
        self.assertTrue(torch.all(result['nir'] == 0))


class SEOManifestNIRTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = config_for(self.root)
        self.rows = []
        statuses = ('original_rgb_ready', 'rgb_ready', 'candidate_rgb', 'pan_only', 'candidate_pan', 'rejected')
        for group_index in range(5):
            group = f'JAX_{group_index}'
            image_dir = self.root / 'pansharpened_he' / group
            mask_dir = self.root / 'shadows_max' / 'shadow_max_compressed' / 'shadow_masks' / group
            nir_dir = self.root / 'nir_aligned_v1' / group
            for folder in (image_dir, mask_dir, nir_dir):
                folder.mkdir(parents=True)
            for view, status in enumerate(statuses):
                stem = f'{group}_{view}'
                image, mask, _, valid = sample_arrays(7, 9)
                Image.fromarray(image).save(image_dir / f'{stem}_rgb.png')
                Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(mask_dir / f'{stem}_shadow.png')
                nir_path, valid_path = nir_dir / f'{stem}_nir.tif', nir_dir / f'{stem}_valid.png'
                Image.fromarray(400 + image[:, :, 0].astype(np.float32) * 4).save(nir_path)
                Image.fromarray(valid.astype(np.uint8) * 255).save(valid_path)
                self.rows.append({'key': group + '/' + stem, 'status': status,
                                  'reference_grid': 'PAN' if status in {'pan_only', 'candidate_pan'} else 'RGB',
                                  'nir': str(nir_path), 'valid': str(valid_path)})
        self.manifest = self.root / 'nir_aligned_v1' / 'registration_with_candidates.csv'
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, ('key', 'status', 'reference_grid', 'nir', 'valid'))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_manifest_filters_after_original_aoi_split_without_moving_membership(self):
        old_config = copy.deepcopy(self.config)
        old_config['dataset']['nir']['enabled'] = False
        accepted = {row['key'] for row in self.rows if row['status'] in {'original_rgb_ready', 'rgb_ready'}}
        datasets = {}
        for split in ('train', 'val'):
            old = get_seo_dataset(old_config, split)
            new = get_seo_dataset(self.config, split)
            expected = [path for path in old.img_paths
                        if Path(path).parent.name + '/' + Path(path).name.removesuffix('_rgb.png') in accepted]
            self.assertEqual(new.img_paths, expected)
            self.assertEqual(new.nir_filter_report['before'], len(old))
            self.assertEqual(new.nir_filter_report['after'], len(new))
            self.assertEqual(len(new.nir_paths), len(new))
            self.assertEqual(len(new.valid_paths), len(new))
            datasets[split] = new
        self.assertFalse({Path(path).parent.name for path in datasets['train'].img_paths}
                         & {Path(path).parent.name for path in datasets['val'].img_paths})

    def test_candidates_are_opt_in_and_pan_results_are_always_excluded(self):
        strict = get_seo_dataset(self.config, 'val')
        self.config['dataset']['nir']['include_candidates'] = True
        candidates = get_seo_dataset(self.config, 'val')
        self.assertEqual(len(candidates), len(strict) * 3 // 2)
        statuses = {row['key']: row['status'] for row in self.rows}
        for path in candidates.img_paths:
            key = Path(path).parent.name + '/' + Path(path).name.removesuffix('_rgb.png')
            self.assertIn(statuses[key], {'original_rgb_ready', 'rgb_ready', 'candidate_rgb'})

    def test_single_image_resolution_uses_same_manifest_eligibility(self):
        for row in self.rows[:6]:
            image = self.root / 'pansharpened_he' / (row['key'] + '_rgb.png')
            if row['status'] in {'original_rgb_ready', 'rgb_ready'}:
                self.assertEqual(resolve_seo_nir_paths(self.config, image), (row['nir'], row['valid']))
            else:
                with self.assertRaisesRegex(ValueError, 'no selected RGB-grid registration'):
                    resolve_seo_nir_paths(self.config, image)
        self.config['dataset']['nir']['include_candidates'] = True
        row = self.rows[2]
        image = self.root / 'pansharpened_he' / (row['key'] + '_rgb.png')
        self.assertEqual(resolve_seo_nir_paths(self.config, image), (row['nir'], row['valid']))

    def test_optional_loading_returns_four_tensors_and_disabled_path_stays_tuple(self):
        dataset = get_seo_dataset(self.config, 'val')
        result = dataset[0]
        self.assertEqual(set(result), {'image', 'mask', 'nir', 'nir_valid'})
        self.assertEqual(tuple(result['image'].shape), (3, 7, 9))
        self.assertEqual(tuple(result['nir'].shape), (1, 7, 9))
        self.assertEqual(result['nir'].dtype, torch.float32)
        self.assertEqual(result['nir_valid'].dtype, torch.bool)
        self.assertEqual(result['mask'].dtype, torch.int64)
        self.config['dataset']['nir']['enabled'] = False
        old = get_seo_dataset(self.config, 'val')
        self.assertIsInstance(old[0], tuple)

    def test_external_rgb_only_transform_is_rejected_but_explicit_joint_val_is_supported(self):
        with self.assertRaisesRegex(ValueError, 'get_nir_transforms'):
            get_seo_dataset(self.config, 'val', transform=get_transforms(self.config, 'val'))
        dataset = get_seo_dataset(self.config, 'train', transform=get_nir_transforms(self.config, 'val'))
        self.assertEqual(tuple(dataset[0]['image'].shape), (3, 7, 9))

    def test_duplicate_manifest_and_missing_eligible_files_fail_clearly(self):
        self.rows.append(self.rows[0].copy())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            get_seo_dataset(self.config, 'val')
        self.rows.pop()
        self.write_manifest()
        selected = get_seo_dataset(self.config, 'val')
        Path(selected.nir_paths[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing eligible'):
            get_seo_dataset(self.config, 'val')


if __name__ == '__main__':
    unittest.main()

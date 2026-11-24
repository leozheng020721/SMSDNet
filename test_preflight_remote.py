"""Portable-bundle preflight checks using tiny sibling SEO/segformer folders."""

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
import yaml

import preflight_remote


class RemotePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / 'segformer'
        self.code.mkdir()
        self.data = self.root / 'SEO'
        self.config_path = self.code / 'config_seo_cca_nir_server.yaml'
        self.manifest = self.data / 'nir_aligned_v1' / 'registration_with_candidates.csv'
        self.rows = []
        height, width = 9, 11
        yy, xx = np.indices((height, width))
        for group_index in range(4):
            group = f'JAX_{group_index}'
            folders = {
                'rgb': self.data / 'pansharpened_he' / group,
                'max': self.data / 'shadows_max' / 'shadow_max_compressed' / 'shadow_masks' / group,
                'min': self.data / 'shadows_min' / 'shadows' / 'shadow_masks' / group,
                'uncertainty': self.data / 'shadows_min' / 'shadows' / 'uncertainty_masks' / group,
                'nir': self.data / 'nir_aligned_v1' / group,
            }
            for folder in folders.values():
                folder.mkdir(parents=True)
            for index in range(2):
                stem = f'{group}_{index}'
                rgb = np.stack(((yy * 17) % 255, (xx * 11) % 255, (yy + xx) * 7), axis=-1).astype(np.uint8)
                mask = np.where((yy + xx) % 2, 0, 255).astype(np.uint8)
                uncertainty = np.zeros_like(mask)
                uncertainty[1, 1] = 1
                nir = (500 + yy * 4 + xx * 3).astype(np.float32)
                nir[0, 0] = np.nan
                valid = np.isfinite(nir)
                for folder in ('max', 'min'):
                    Image.fromarray(mask).save(folders[folder] / (stem + '_shadow.png'))
                Image.fromarray(rgb).save(folders['rgb'] / (stem + '_rgb.png'))
                Image.fromarray(uncertainty).save(folders['uncertainty'] / (stem + '_uncertainty_mask.png'))
                Image.fromarray(nir).save(folders['nir'] / (stem + '_nir.tif'))
                Image.fromarray(valid.astype(np.uint8) * 255).save(folders['nir'] / (stem + '_valid.png'))
                self.rows.append({'key': group + '/' + stem, 'status': 'original_rgb_ready',
                                  'reference_grid': 'RGB', 'nir': group + '/' + stem + '_nir.tif',
                                  'valid': group + '/' + stem + '_valid.png'})
        self.write_manifest()
        self.config = {
            'random_seed': 42, 'device': 'cuda',
            'dataset': {
                'type': 'seo', 'root_dir': '../SEO', 'val_ratio': 0.25,
                'spatial_mode': 'crop', 'input_size': 8,
                'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
                'supervision': {'mode': 'minmax_agreement', 'missing_policy': 'error'},
                'nir': {'enabled': True, 'root_dir': '../SEO/nir_aligned_v1',
                        'manifest': 'registration_with_candidates.csv', 'include_candidates': False,
                        'normalization': {'percentiles': [0, 100]}},
            },
            'model': {'ignore_index': 255},
        }
        self.config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        previous = Path.cwd()
        os.chdir(self.code)
        self.addCleanup(os.chdir, previous)

    def write_manifest(self):
        with self.manifest.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, ('key', 'status', 'reference_grid', 'nir', 'valid'))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_sibling_layout_reads_two_samples_without_model_checkpoint_or_cuda(self):
        import dataset
        import model_factory

        checkpoint = self.code / 'checkpoint.pth'
        checkpoint.write_bytes(b'This file is intentionally not a serialized checkpoint.')
        original = dataset.SEONIRDataset.__getitem__
        sampled = []

        def inspect_sample(source, index):
            sampled.append((source.split if hasattr(source, 'split') else source.transform.split, index))
            return original(source, index)

        with mock.patch.object(dataset.SEONIRDataset, '__getitem__', autospec=True, side_effect=inspect_sample), \
                mock.patch.object(torch, 'load', side_effect=AssertionError('must not load checkpoints')), \
                mock.patch.object(model_factory, 'build_model', side_effect=AssertionError('must not build models')), \
                mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU check not requested')):
            report = preflight_remote.run_preflight(self.config_path, checkpoint='checkpoint.pth')
        self.assertTrue(report['ok'])
        self.assertTrue(report['aoi_disjoint'])
        self.assertEqual(report['manifest']['eligible_rows'], 8)
        self.assertFalse(report['checkpoint']['loaded'])
        self.assertFalse(report['environment']['gpu_checked'])
        self.assertEqual(len(sampled), 2)
        self.assertTrue(all(index == 0 for _, index in sampled))
        self.assertEqual((report['splits']['train']['images'], report['splits']['val']['images']), (6, 2))
        for split in ('train', 'val'):
            sample = report['splits'][split]['sample']
            self.assertEqual(sample['image_shape'], [3, 9, 11])
            self.assertEqual(sample['nir_shape'], [1, 9, 11])
            self.assertTrue(sample['nir_finite'])
            self.assertIn('255', sample['label_counts'])
            self.assertIn('min_mask', sample['sources'])
            self.assertIn('uncertainty_mask', sample['sources'])
        json.dumps(report, allow_nan=False)

    def test_windows_absolute_and_escaping_manifest_paths_fail_before_sampling(self):
        import dataset

        original = self.rows[0]['nir']
        paths = (r'E:\Dataset\SEO\nir_aligned_v1\JAX_0\JAX_0_0_nir.tif',
                 '/data/SEO/nir_aligned_v1/JAX_0/JAX_0_0_nir.tif', '../outside.tif')
        for path in paths:
            with self.subTest(path=path):
                self.rows[0]['nir'] = path
                self.write_manifest()
                with mock.patch.object(dataset.SEONIRDataset, '__getitem__') as read:
                    with self.assertRaisesRegex(ValueError, 'relative POSIX'):
                        preflight_remote.run_preflight(self.config_path)
                read.assert_not_called()
        self.rows[0]['nir'] = original

    def test_minmax_shape_mismatch_is_detected_by_the_real_sample_load(self):
        for path in (self.data / 'shadows_min' / 'shadows' / 'uncertainty_masks').rglob('*.png'):
            Image.fromarray(np.zeros((1, 1), np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'spatial sizes differ'):
            preflight_remote.run_preflight(self.config_path)

    def test_overlapping_aois_are_rejected_before_reading_pixels(self):
        import dataset

        class OverlappingDataset:
            img_paths = ['../SEO/pansharpened_he/JAX_0/JAX_0_0_rgb.png']

            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise AssertionError('overlapping data must not be sampled')

        with mock.patch.object(dataset, 'get_seo_dataset', return_value=OverlappingDataset()):
            with self.assertRaisesRegex(ValueError, 'AOIs overlap'):
                preflight_remote.run_preflight(self.config_path)

    def test_missing_checkpoint_fails_without_deserializing(self):
        with mock.patch.object(torch, 'load', side_effect=AssertionError('must not load checkpoints')):
            with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint file'):
                preflight_remote.run_preflight(self.config_path, checkpoint='missing.pth')

    def test_bundle_counts_and_ordered_keys_are_checked_before_sampling(self):
        import dataset

        counts, hashes = {}, {}
        for split in ('train', 'val'):
            source = dataset.get_seo_dataset(self.config, split)
            keys = [Path(path).parent.name + '/' + Path(path).name.removesuffix('_rgb.png')
                    for path in source.img_paths]
            counts[split] = len(source)
            hashes[split] = hashlib.sha256(('\n'.join(keys) + '\n').encode()).hexdigest()
        path = self.data / 'BUNDLE_INFO.json'
        path.write_text(json.dumps({'expected_counts': counts, 'split_key_sha256': hashes}), encoding='utf-8')
        report = preflight_remote.run_preflight(self.config_path)
        self.assertTrue(report['bundle_info']['validated'])
        self.assertEqual(report['split_key_sha256'], hashes)
        for bad_counts, bad_hashes in (({**counts, 'train': counts['train'] + 1}, hashes),
                                      (counts, {**hashes, 'val': '0' * 64})):
            path.write_text(json.dumps({'expected_counts': bad_counts, 'split_key_sha256': bad_hashes}), encoding='utf-8')
            with self.subTest(counts=bad_counts, hashes=bad_hashes), \
                    mock.patch.object(dataset.SEONIRDataset, '__getitem__') as read:
                with self.assertRaisesRegex(ValueError, 'BUNDLE_INFO.json'):
                    preflight_remote.run_preflight(self.config_path)
                read.assert_not_called()

    def test_requested_gpu_check_reports_unavailable_cuda_without_allocating(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.object(torch.cuda, 'get_device_properties') as query:
            with self.assertRaisesRegex(ValueError, 'cuda.is_available'):
                preflight_remote.run_preflight(self.config_path, check_gpu=True)
        query.assert_not_called()

    def test_cli_failure_includes_working_directory_hint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = preflight_remote.main(['--config', 'missing.yaml'])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 2)
        self.assertFalse(report['ok'])
        self.assertIn('segformer', report['run_hint'])
        self.assertIn('../SEO', report['run_hint'])


if __name__ == '__main__':
    unittest.main()

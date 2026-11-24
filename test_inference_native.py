"""CPU integration checks for native-resolution SEO inference entry points."""

import contextlib
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
import torch
from torch.utils.data import Subset
import yaml

import inference

try:
    import onnxruntime
except ModuleNotFoundError:
    aisd_inference = None
else:
    import aisd_inference


class PixelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, images):
        self.batches.append(tuple(images.shape))
        red = images[:, :1] * 0.229 + 0.485
        return torch.cat((0.5 - red, red - 0.5), dim=1)


class FakeSession:
    def __init__(self):
        self.batches = []

    def get_inputs(self):
        return [SimpleNamespace(name='input')]

    def get_providers(self):
        return ['CPUExecutionProvider']

    def run(self, outputs, inputs):
        images = inputs['input']
        self.batches.append(images.shape)
        red = images[:, :1] * 0.229 + 0.485
        return [np.concatenate((0.5 - red, red - 0.5), axis=1)]


class NativeInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix='seo_native_')
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = {
            'random_seed': 42,
            'num_workers': 0,
            'pin_memory': False,
            'dataset': {
                'type': 'seo',
                'spatial_mode': 'crop',
                'input_size': 16,
                'tile_stride': 12,
                'tile_batch_size': 3,
                'num_classes': 2,
                'in_channels': 3,
                'root_dir': str(self.root),
                'val_ratio': 0.5,
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'div_value': 0.3,
                },
            },
        }
        self.image = np.zeros((21, 39, 3), dtype=np.uint8)
        self.image[:, :, 2] = 64
        self.image[:10, :, 2] = 192
        self.expected = np.zeros((21, 39), dtype=np.uint8)
        self.expected[:10] = 1

    def assert_tile_batches(self, batches):
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(shape[0] <= 3 and shape[-2:] == (16, 16) for shape in batches))

    def write_seo_dataset(self):
        expected_by_path = {}
        for index in range(4):
            group = f'JAX_{100 + index}'
            image_dir = self.root / 'pansharpened_he' / group
            mask_dir = self.root / 'shadows_max' / 'shadow_max_compressed' / 'shadow_masks' / group
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            image = np.roll(self.image, index * 3, axis=0)
            expected = np.roll(self.expected, index * 3, axis=0)
            image_path = image_dir / 'view_rgb.png'
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_dir / 'view_shadow.png'), (1 - expected) * 255))
            expected_by_path[str(image_path.resolve())] = expected
        return expected_by_path

    def test_single_image_preserves_rgb_scale_and_native_shape(self):
        image_path = self.root / 'JAX_100_001_rgb.png'
        self.assertTrue(cv2.imwrite(str(image_path), self.image))
        model = PixelModel()
        prediction = inference.inference_single_image(
            model, str(image_path), self.config, torch.device('cpu')
        )
        np.testing.assert_array_equal(prediction, self.expected)
        self.assert_tile_batches(model.batches)

    @unittest.skipIf(aisd_inference is None, 'onnxruntime is an optional dependency')
    def test_onnx_matches_seo_rgb_image_to_shadow_mask(self):
        image_dir = self.root / 'images'
        mask_dir = self.root / 'masks'
        output_dir = self.root / 'output'
        image_dir.mkdir()
        mask_dir.mkdir()
        self.assertTrue(cv2.imwrite(str(image_dir / 'JAX_100_001_rgb.png'), self.image))
        self.assertTrue(cv2.imwrite(
            str(mask_dir / 'JAX_100_001_shadow.png'), (1 - self.expected) * 255
        ))
        session = FakeSession()
        with patch.object(aisd_inference.onnxruntime, 'InferenceSession', return_value=session):
            with patch.object(aisd_inference.onnxruntime, 'get_available_providers',
                              return_value=['CPUExecutionProvider']):
                with patch.object(aisd_inference, 'tqdm', side_effect=lambda values, **kwargs: values):
                    with contextlib.redirect_stdout(io.StringIO()):
                        metrics = aisd_inference.batch_infer(
                            'fake.onnx', str(image_dir), str(output_dir), self.config,
                            mask_dir=str(mask_dir),
                        )
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['ber'], 0.0)
        saved_mask = cv2.imread(
            str(output_dir / 'JAX_100_001_rgb_mask.png'), cv2.IMREAD_GRAYSCALE
        )
        np.testing.assert_array_equal(saved_mask, self.expected * 255)
        self.assert_tile_batches(session.batches)

    def test_manifest_preserves_subset_indices_and_disambiguates_matching_aoi_filenames(self):
        self.write_seo_dataset()
        dataset = inference.build_inference_dataset(self.config, split='val')
        self.assertEqual(len(dataset), 2)
        subset = Subset(dataset, [1, 0])
        output_dir = self.root / 'manifest_output'
        records, manifest_path = inference.export_dataset_manifest(
            subset, output_dir, split='val', num_classes=2, group_by_parent=True,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record['prediction_path'] for record in records}), 2)
        for record, original_index in zip(records, [1, 0]):
            image_path = Path(dataset.img_paths[original_index]).resolve()
            self.assertEqual(int(record['dataset_index']), original_index)
            self.assertEqual(Path(record['image_path']), image_path)
            self.assertEqual(Path(record['mask_path']), Path(dataset.mask_paths[original_index]).resolve())
            self.assertEqual(record['group'], image_path.parent.name)
            for key, suffix in (('prediction_path', '_pred.png'), ('binary_path', '_shadow.png'),
                                ('visualization_path', '_vis.png')):
                path = Path(record[key])
                self.assertTrue(path.is_absolute())
                self.assertEqual(path, output_dir.resolve() / image_path.parent.name / (image_path.stem + suffix))
        with Path(manifest_path).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['dataset_index']) for row in rows], [1, 0])
        self.assertEqual([row['image_path'] for row in rows], [record['image_path'] for record in records])

    def test_dataset_exports_native_class_ids_and_white_shadow_masks_per_aoi(self):
        expected_by_path = self.write_seo_dataset()
        output_dir = self.root / 'predictions'
        model = PixelModel()
        with patch.object(inference, 'visualize_prediction') as visualize, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metrics = inference.inference_dataset(
                model, self.config, torch.device('cpu'), split='val', output_dir=str(output_dir),
            )
        self.assertEqual(metrics['ber'], 0.0)
        self.assertEqual(visualize.call_count, 2)
        with (output_dir / 'val_manifest.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['prediction_path'] for row in rows}), 2)
        for row in rows:
            expected = expected_by_path[row['image_path']]
            raw = cv2.imread(row['prediction_path'], cv2.IMREAD_GRAYSCALE)
            binary = cv2.imread(row['binary_path'], cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(raw, expected)
            np.testing.assert_array_equal(binary, expected * 255)
            self.assertEqual(raw.shape, expected.shape)
            self.assertEqual(set(np.unique(raw)), {0, 1})
            self.assertEqual(set(np.unique(binary)), {0, 255})
        self.assert_tile_batches(model.batches)

    def test_export_only_cli_writes_validation_manifest_without_building_model(self):
        self.write_seo_dataset()
        self.config['device'] = 'cuda'
        config_path = self.root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        output_dir = self.root / 'export_only'
        checkpoint = {'config': {'dataset': {'spatial_mode': 'crop', 'input_size': 16}}}
        arguments = ['inference.py', '--config', str(config_path), '--checkpoint', 'unused.pth',
                     '--split', 'val', '--output', str(output_dir), '--export-only']
        with patch.object(sys, 'argv', arguments), \
                patch.object(inference.torch, 'load', return_value=checkpoint), \
                patch.object(inference, 'build_model') as build_model, \
                contextlib.redirect_stdout(io.StringIO()):
            inference.main()
        build_model.assert_not_called()
        with (output_dir / 'val_manifest.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(output_dir.rglob('*_pred.png')))


if __name__ == '__main__':
    unittest.main()

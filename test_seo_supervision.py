"""Regression checks for SEO Min/Max agreement and official uncertainty filtering."""

import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch

from dataset import AISDDataset, get_nir_transforms, get_seo_dataset, get_transforms
from metrics import fast_hist, get_metrics
from model_factory import resolve_checkpoint_config
import test_dataset_nir as nir_fixture


class SEOSupervisionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = nir_fixture.config_for(self.root, size=8)
        self.config['dataset']['nir']['enabled'] = False
        self.config['dataset']['supervision'] = {'mode': 'minmax_agreement'}
        group = 'JAX_100'
        self.image_path = self.root / 'pansharpened_he' / group / 'view_rgb.png'
        self.mask_path = self.root / 'shadows_max/shadow_max_compressed/shadow_masks' / group / 'view_shadow.png'
        self.min_path = self.root / 'shadows_min/shadows/shadow_masks' / group / 'view_shadow.png'
        self.uncertainty_path = (self.root / 'shadows_min/shadows/uncertainty_masks'
                                 / group / 'view_uncertainty_mask.png')
        maximum = np.uint8([[0, 255, 0, 255], [255, 0, 255, 0], [0, 255, 0, 255]])
        minimum = maximum.copy()
        minimum[0, 1], minimum[1, 0] = 0, 0
        uncertainty = np.zeros_like(maximum)
        uncertainty[0, 2], uncertainty[1, 3] = 255, 1
        image = np.repeat(np.uint8([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])[..., None], 3, 2)
        for path, values in ((self.image_path, image), (self.mask_path, maximum),
                             (self.min_path, minimum), (self.uncertainty_path, uncertainty)):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(path), values))
        self.expected = (maximum < 127).astype(np.int64)
        self.expected[(minimum != maximum) | (uncertainty != 0)] = 255

    def dataset(self, config=None, transform=None):
        return AISDDataset([str(self.image_path)], [str(self.mask_path)],
                           self.config if config is None else config, transform)

    def test_agreement_keeps_both_classes_and_excludes_nonzero_min_uncertainty(self):
        # Deliberately conflicting Max-side uncertainty must not alter the official Min-side rule.
        max_uncertainty = self.root / 'shadows_max/shadow_max_compressed/uncertainty_masks/JAX_100/view_uncertainty_mask.png'
        max_uncertainty.parent.mkdir(parents=True)
        cv2.imwrite(str(max_uncertainty), np.full(self.expected.shape, 255, np.uint8))
        _, target = self.dataset()[0]
        np.testing.assert_array_equal(target, self.expected)
        self.assertEqual(set(target.unique().tolist()), {0, 1, 255})
        # Predictions only fail on excluded pixels, so filtered BER must be zero.
        prediction = target.clone()
        prediction[target == 255] = 0
        hist = fast_hist(prediction, target, 2)
        self.assertEqual(hist.sum(), int((target != 255).sum()))
        self.assertEqual(get_metrics(hist)['ber'], 0)

    def test_uncertainty_is_optional_but_min_max_disagreement_is_always_ignored(self):
        self.config['dataset']['supervision']['ignore_uncertainty'] = False
        self.uncertainty_path.unlink()
        _, target = self.dataset()[0]
        self.assertEqual(int(target[0, 2]), 1)
        self.assertEqual(int(target[1, 3]), 1)
        self.assertEqual(int(target[0, 1]), 255)
        self.assertEqual(int(target[1, 0]), 255)

    def test_palette_png_indices_are_decoded_before_min_max_comparison(self):
        for path in (self.min_path, self.uncertainty_path):
            grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            indexed = Image.fromarray((grayscale > 0).astype(np.uint8))
            indexed.putpalette([0, 0, 0, 255, 255, 255] + [0] * (254 * 3))
            indexed.save(path)
        _, target = self.dataset()[0]
        np.testing.assert_array_equal(target, self.expected)

    def test_legacy_max_does_not_require_min_files(self):
        self.config['dataset'].pop('supervision')
        self.min_path.unlink()
        self.uncertainty_path.unlink()
        _, target = self.dataset()[0]
        expected = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE) < 127
        np.testing.assert_array_equal(target, expected)

    def test_missing_min_or_uncertainty_fails_before_training_without_dropping_samples(self):
        for path in (self.min_path, self.uncertainty_path):
            saved = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Extract shadows_min.tar.gz'):
                self.dataset()
            path.write_bytes(saved)

    def test_shape_mismatch_is_rejected_instead_of_resizing_labels(self):
        for path in (self.min_path, self.uncertainty_path):
            saved = path.read_bytes()
            cv2.imwrite(str(path), np.zeros((2, 2), np.uint8))
            with self.assertRaisesRegex(ValueError, 'spatial sizes differ'):
                self.dataset()[0]
            path.write_bytes(saved)

    def test_joint_rgb_geometry_preserves_ignore_pixels_and_padding(self):
        transform = get_transforms(self.config, 'train')
        _, target = self.dataset(transform=transform)[0]
        self.assertEqual(tuple(target.shape), (8, 8))
        self.assertEqual(int((target != 255).sum()), int((self.expected != 255).sum()))
        self.assertEqual(int((target == 1).sum()), int((self.expected == 1).sum()))

    def test_resize_affine_does_not_create_background_from_all_ignored_labels(self):
        self.config['dataset'].update(spatial_mode='resize', input_size=64)
        cv2.imwrite(str(self.uncertainty_path), np.full(self.expected.shape, 255, np.uint8))
        transform = get_transforms(self.config, 'train')
        affine = next(operation for operation in transform.transforms
                      if isinstance(operation, A.ShiftScaleRotate))
        affine.p = 1
        for _ in range(3):
            _, target = self.dataset(transform=transform)[0]
            self.assertTrue(torch.all(target == 255))

    def test_nir_geometry_preserves_ignored_labels_and_nir_validity_separately(self):
        transform = get_nir_transforms(self.config, 'val')
        transform.geometry = A.Compose([A.HorizontalFlip(p=1)],
                                        additional_targets={'nir': 'image', 'nir_valid': 'image'})
        image, target = self.dataset()[0]
        result = transform(image=image, mask=target.numpy().astype(np.uint8),
                           nir=np.ones(self.expected.shape, np.float32),
                           nir_valid=np.ones(self.expected.shape, bool))
        np.testing.assert_array_equal(result['mask'], self.expected[:, ::-1])
        self.assertTrue(result['nir_valid'].all())

    def test_all_ignored_nir_labels_remain_ignored_in_train_and_validation(self):
        cv2.imwrite(str(self.uncertainty_path), np.full(self.expected.shape, 255, np.uint8))
        image, target = self.dataset()[0]
        for split in ('train', 'val'):
            with self.subTest(split=split):
                result = get_nir_transforms(self.config, split)(
                    image=image, mask=target.numpy().astype(np.uint8),
                    nir=np.ones(self.expected.shape, np.float32),
                    nir_valid=np.ones(self.expected.shape, bool))
                self.assertTrue(torch.all(result['mask'] == 255))
                self.assertTrue(result['nir_valid'].any())

    def test_invalid_protocol_settings_fail_clearly(self):
        for options in ({'mode': 'min'}, {'mode': 'max', 'ignore_uncertainty': True},
                        {'mode': 'minmax_agreement', 'ignore_uncertainty': 'true'},
                        {'mode': 'minmax_agreement', 'missing_policy': 'silent'}):
            with self.subTest(options=options):
                self.config['dataset']['supervision'] = options
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_checkpoint_protocol_is_restored_unless_runtime_explicitly_overrides_it(self):
        current = copy.deepcopy(self.config)
        current['model'] = {'variant': 'baseline'}
        saved = copy.deepcopy(current)
        current['dataset'].pop('supervision')
        resolved = resolve_checkpoint_config(current, {'config': saved})
        self.assertEqual(resolved['dataset']['supervision'], saved['dataset']['supervision'])
        current['dataset']['supervision'] = {'mode': 'max'}
        resolved = resolve_checkpoint_config(current, {'config': saved})
        self.assertEqual(resolved['dataset']['supervision'], {'mode': 'max'})

    def test_runtime_paths_or_empty_options_preserve_saved_protocol(self):
        saved = copy.deepcopy(self.config)
        saved['model'] = {'variant': 'baseline'}
        for overrides in ({}, {'min_root': 'moved/min'}, {'ignore_uncertainty': False}):
            with self.subTest(overrides=overrides):
                current = copy.deepcopy(saved)
                current['dataset']['supervision'] = overrides
                resolved = resolve_checkpoint_config(current, {'config': saved})
                self.assertEqual(resolved['dataset']['supervision'],
                                 {'mode': 'minmax_agreement', **overrides})


class SEOSupervisionMembershipTest(unittest.TestCase):
    def fixture(self, create_min=True):
        fixture = nir_fixture.SEOManifestNIRTest(methodName='test_optional_loading_returns_four_tensors_and_disabled_path_stays_tuple')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root, config = fixture.root, fixture.config
        if not create_min:
            return root, config
        for maximum in (root / 'shadows_max/shadow_max_compressed/shadow_masks').rglob('*.png'):
            minimum = root / 'shadows_min/shadows/shadow_masks' / maximum.parent.name / maximum.name
            uncertainty = (root / 'shadows_min/shadows/uncertainty_masks' / maximum.parent.name
                           / maximum.name.replace('_shadow.png', '_uncertainty_mask.png'))
            minimum.parent.mkdir(parents=True, exist_ok=True)
            uncertainty.parent.mkdir(parents=True, exist_ok=True)
            label = cv2.imread(str(maximum), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(minimum), label)
            cv2.imwrite(str(uncertainty), np.zeros_like(label))
        return root, config

    def test_rgb_and_nir_splits_keep_exact_image_membership(self):
        root, config = self.fixture()
        for nir in (False, True):
            config['dataset']['nir']['enabled'] = nir
            for split in ('train', 'val'):
                with self.subTest(nir=nir, split=split):
                    transform = get_nir_transforms if nir else get_transforms
                    old = get_seo_dataset(config, split, transform=transform(config, 'val'))
                    new_config = copy.deepcopy(config)
                    new_config['dataset']['supervision'] = {'mode': 'minmax_agreement'}
                    new = get_seo_dataset(new_config, split, transform=transform(new_config, 'val'))
                    self.assertEqual(old.img_paths, new.img_paths)
                    self.assertEqual(old.mask_paths, new.mask_paths)
                    old_item, new_item = old[0], new[0]
                    self.assertEqual(type(old_item), type(new_item))
                    if nir:
                        torch.testing.assert_close(old_item['mask'], new_item['mask'])

    def test_skip_missing_pairs_reports_counts_without_changing_aoi_split(self):
        for nir in (False, True):
            with self.subTest(nir=nir):
                root, config = self.fixture()
                config['dataset']['nir']['enabled'] = nir
                old_val = get_seo_dataset(config, 'val')
                old_train = get_seo_dataset(config, 'train')
                absent = Path(old_val.mask_paths[0])
                (root / 'shadows_min/shadows/shadow_masks' / absent.parent.name / absent.name).unlink()
                config['dataset']['supervision'] = {'mode': 'minmax_agreement', 'missing_policy': 'skip'}
                new_val = get_seo_dataset(config, 'val')
                new_train = get_seo_dataset(config, 'train')
                self.assertEqual(new_val.img_paths, old_val.img_paths[1:])
                self.assertEqual(new_train.img_paths, old_train.img_paths)
                report = new_val.supervision_filter_report
                self.assertEqual((report['before'], report['after'], report['missing']),
                                 (len(old_val), len(old_val) - 1, 1))
                self.assertEqual(report['missing_policy'], 'skip')
                self.assertEqual(len(report['missing_examples']), 1)
                self.assertEqual(new_train.supervision_filter_report['missing'], 0)

    def test_skip_mode_still_requires_extraction_and_never_returns_an_empty_split(self):
        root, config = self.fixture(create_min=False)
        config['dataset']['supervision'] = {'mode': 'minmax_agreement', 'missing_policy': 'skip'}
        with self.assertRaisesRegex(FileNotFoundError, 'Extract shadows_min.tar.gz'):
            get_seo_dataset(config, 'val')
        for folder in ('shadow_masks', 'uncertainty_masks'):
            (root / 'shadows_min/shadows' / folder).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'No complete Min/Max'):
            get_seo_dataset(config, 'val')


if __name__ == '__main__':
    unittest.main()

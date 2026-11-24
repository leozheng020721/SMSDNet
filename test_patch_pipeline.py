"""CPU regression checks for native-scale crops and stitched predictions."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
import cv2
import numpy as np
import torch

from dataset import AISDDataset, get_aisd_loaders, get_seo_dataset, get_seo_loaders, get_transforms
from patch_inference import is_crop_mode, sliding_window_logits


def crop_config():
    return {
        'dataset': {
            'type': 'aisd', 'spatial_mode': 'crop', 'input_size': 512,
            'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
        },
        'training': {'batch_size': 2}, 'num_workers': 0, 'pin_memory': False,
    }


class CropTransformTest(unittest.TestCase):
    @staticmethod
    def geometry_only(transform):
        for operation in transform.transforms:
            if isinstance(operation, (A.HorizontalFlip, A.VerticalFlip, A.RandomRotate90, A.RandomBrightnessContrast)):
                operation.p = 0
        return transform

    def test_train_crops_original_pixels_and_keeps_mask_aligned(self):
        rows, columns = np.indices((633, 719))
        mask = ((rows * 3 + columns * 7) % 251).astype(np.uint8)
        image = np.stack([mask, rows % 251, columns % 251], axis=-1).astype(np.uint8)
        transform = self.geometry_only(get_transforms(crop_config(), 'train'))
        self.assertFalse(any(isinstance(operation, (A.Resize, A.ShiftScaleRotate)) for operation in transform.transforms))
        result = transform(image=image, mask=mask)
        self.assertEqual(tuple(result['image'].shape), (3, 512, 512))
        decoded = torch.round(result['image'] * 255).to(torch.int64)
        torch.testing.assert_close(decoded[0], result['mask'].to(torch.int64))
        # Adjacent coordinates stay one original pixel apart, modulo the encoding.
        torch.testing.assert_close((decoded[1, 1:] - decoded[1, :-1]) % 251, torch.ones((511, 512), dtype=torch.int64))
        torch.testing.assert_close((decoded[2, :, 1:] - decoded[2, :, :-1]) % 251, torch.ones((512, 511), dtype=torch.int64))

    def test_small_train_image_uses_ignored_padding_without_scaling(self):
        transform = self.geometry_only(get_transforms(crop_config(), 'train'))
        image = np.full((97, 143, 3), 19, dtype=np.uint8)
        mask = np.full((97, 143), 3, dtype=np.uint8)
        result = transform(image=image, mask=mask)
        valid = result['mask'] != 255
        self.assertEqual(tuple(result['mask'].shape), (512, 512))
        self.assertEqual(int(valid.sum()), 97 * 143)
        self.assertTrue(torch.all(result['mask'][valid] == 3))
        torch.testing.assert_close(result['image'][:, valid], torch.full((3, 97 * 143), 19 / 255))
        self.assertTrue(torch.all(result['image'][:, ~valid] == 0))

    def test_validation_preserves_original_dimensions_and_pixels(self):
        image = np.random.default_rng(7).integers(0, 256, size=(347, 619, 3), dtype=np.uint8)
        mask = image[:, :, 0].copy()
        result = get_transforms(crop_config(), 'val')(image=image, mask=mask)
        self.assertEqual(tuple(result['image'].shape), (3, 347, 619))
        np.testing.assert_array_equal(result['mask'].numpy(), mask)
        np.testing.assert_array_equal(torch.round(result['image'] * 255).byte().permute(1, 2, 0).numpy(), image)

    def test_dataset_rejects_misaligned_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, mask_path = root / 'image.png', root / 'mask.png'
            cv2.imwrite(str(image_path), np.zeros((17, 23, 3), dtype=np.uint8))
            cv2.imwrite(str(mask_path), np.zeros((17, 22), dtype=np.uint8))
            dataset = AISDDataset([str(image_path)], [str(mask_path)], crop_config())
            with self.assertRaisesRegex(ValueError, 'spatial sizes differ'):
                dataset[0]

    def test_validation_loader_batches_one_variable_size_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                (root / split / 'shadow').mkdir(parents=True)
                (root / split / 'mask').mkdir(parents=True)
                for index, shape in enumerate(((37, 53), (43, 59))):
                    cv2.imwrite(str(root / split / 'shadow' / f'{index}.png'), np.zeros((*shape, 3), dtype=np.uint8))
                    cv2.imwrite(str(root / split / 'mask' / f'{index}.png'), np.zeros(shape, dtype=np.uint8))
            config = crop_config()
            config['dataset'].update(root_dir=str(root), split_mode='predefined', train_dir='train', val_dir='val')
            train_loader, val_loader = get_aisd_loaders(config)
            self.assertEqual(train_loader.batch_size, 2)
            self.assertEqual(val_loader.batch_size, 1)
            self.assertEqual([tuple(images.shape[-2:]) for images, _ in val_loader], [(37, 53), (43, 59)])

    def test_default_mode_stays_resize_and_test_split_is_explicit(self):
        config = crop_config()
        config['dataset'].pop('spatial_mode')
        self.assertFalse(is_crop_mode(config))
        self.assertIsInstance(get_transforms(config, 'val').transforms[0], A.Resize)
        with self.assertRaisesRegex(ValueError, 'independent test split'):
            get_seo_dataset(config, 'test')

    def test_seo_grouped_split_is_shared_and_validation_uses_original_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for group_index in range(3):
                group = f'JAX_{group_index}'
                image_dir = root / 'pansharpened_he' / group
                mask_dir = root / 'shadows_max' / 'shadow_max_compressed' / 'shadow_masks' / group
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                for view in range(2):
                    shape = (31 + view, 47 + view)
                    cv2.imwrite(str(image_dir / f'view{view}_rgb.png'), np.zeros((*shape, 3), dtype=np.uint8))
                    cv2.imwrite(str(mask_dir / f'view{view}_shadow.png'), np.full(shape, 255, dtype=np.uint8))
            config = crop_config()
            config['random_seed'] = 42
            config['dataset'].update(type='seo', root_dir=str(root), val_ratio=1 / 3)
            train_loader, val_loader = get_seo_loaders(config)
            standalone = get_seo_dataset(config, 'val')
            train_groups = {Path(path).parent.name for path in train_loader.dataset.img_paths}
            val_groups = {Path(path).parent.name for path in val_loader.dataset.img_paths}
            self.assertFalse(train_groups & val_groups)
            self.assertEqual(standalone.img_paths, val_loader.dataset.img_paths)
            self.assertEqual(val_loader.batch_size, 1)
            self.assertEqual([tuple(images.shape[-2:]) for images, _ in val_loader], [(31, 47), (32, 48)])


class SlidingWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_identity_reconstructs_original_coordinates_for_multiple_sizes(self):
        class IdentityLogits(torch.nn.Module):
            def forward(self, inputs):
                return torch.cat([inputs[:, :1], -inputs[:, :1]], dim=1)

        for height, width in ((31, 47), (512, 512), (701, 619), (1025, 257)):
            with self.subTest(shape=(height, width)):
                inputs = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
                result = sliding_window_logits(IdentityLogits(), inputs, 'cpu', tile_size=512, stride=384, tile_batch_size=2)
                self.assertEqual(result.device.type, 'cpu')
                self.assertEqual(tuple(result.shape), (1, 2, height, width))
                torch.testing.assert_close(result[:, :1], inputs, rtol=0, atol=0)
                torch.testing.assert_close(result[:, 1:], -inputs, rtol=0, atol=0)

    def test_overlap_averages_logits_before_classification(self):
        class TopLeftLogits(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, :, :1, :1].expand(-1, -1, 4, 4)

        rows, columns = torch.meshgrid(torch.arange(6), torch.arange(7), indexing='ij')
        inputs = (10 * rows + columns).float()[None, None]
        result = sliding_window_logits(TopLeftLogits(), inputs, 'cpu', tile_size=4, stride=3, tile_batch_size=3)
        expected = torch.tensor([
            [0, 0, 0, 1.5, 3, 3, 3],
            [0, 0, 0, 1.5, 3, 3, 3],
            [10, 10, 10, 11.5, 13, 13, 13],
            [10, 10, 10, 11.5, 13, 13, 13],
            [20, 20, 20, 21.5, 23, 23, 23],
            [20, 20, 20, 21.5, 23, 23, 23],
        ])[None, None]
        torch.testing.assert_close(result, expected)

    def test_invalid_stride_batch_and_outputs_fail_clearly(self):
        inputs = torch.ones(1, 3, 13, 17)
        for stride in (0, -1, 513):
            with self.subTest(stride=stride), self.assertRaisesRegex(ValueError, 'stride'):
                sliding_window_logits(torch.nn.Identity(), inputs, 'cpu', stride=stride)
        with self.assertRaisesRegex(ValueError, 'B=1'):
            sliding_window_logits(torch.nn.Identity(), inputs.repeat(2, 1, 1, 1), 'cpu')
        with self.assertRaisesRegex(ValueError, 'tile_batch_size'):
            sliding_window_logits(torch.nn.Identity(), inputs, 'cpu', tile_batch_size=0)
        with self.assertRaisesRegex(ValueError, 'preserve'):
            sliding_window_logits(torch.nn.AvgPool2d(2), inputs, 'cpu')


if __name__ == '__main__':
    unittest.main()

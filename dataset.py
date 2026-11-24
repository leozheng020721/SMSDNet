import os
import inspect
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import time
import warnings
import cv2

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from patch_inference import is_crop_mode

try:
    import netCDF4
except ImportError:
    netCDF4 = None

class KappaSetDataset(Dataset):
    """
    针对KappaSet数据集的PyTorch Dataset实现。
    直接从NetCDF文件读取RGB波段(B02/B03/B04)和Label。
    """
    def __init__(self, root_dir, split='train', config=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.config = config
        self.transform = transform
        
        # KappaSet数据集组织结构：kappaset/<month>/*.nc 和 kappaset/test/*.nc
        # 我们需要收集所有NetCDF文件并划分为train/val
        
        if split == 'test':
            # 测试集在单独的test文件夹
            test_dir = os.path.join(root_dir, 'kappaset', 'test')
            self.nc_files = sorted(glob(os.path.join(test_dir, '*.nc')))
        else:
            # 训练集和验证集从月份文件夹中收集
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            all_files = []
            for month in months:
                month_dir = os.path.join(root_dir, 'kappaset', month)
                if os.path.exists(month_dir):
                    month_files = glob(os.path.join(month_dir, '*.nc'))
                    all_files.extend(month_files)
            
            all_files = sorted(all_files)
            
            # 按照80/20划分训练集和验证集
            num_files = len(all_files)
            split_idx = int(num_files * 0.8)
            
            if split == 'train':
                self.nc_files = all_files[:split_idx]
            elif split == 'val':
                self.nc_files = all_files[split_idx:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        if len(self.nc_files) == 0:
            raise ValueError(f"No NetCDF files found for split '{split}' in {root_dir}")
        
        print(f"[{split.upper()}] Loaded {len(self.nc_files)} NetCDF files")

    def __len__(self):
        return len(self.nc_files)

    def __getitem__(self, idx):
        if netCDF4 is None:
            raise ImportError("KappaSetDataset requires the netCDF4 package")

        nc_path = self.nc_files[idx]
        
        # 添加重试机制处理HDF错误（多进程并发访问问题）
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                # 打开NetCDF文件
                with netCDF4.Dataset(nc_path, 'r') as nc:
                    # 提取RGB波段
                    # 根据Sentinel-2波段定义：
                    # B02 (490 nm) - Blue
                    # B03 (560 nm) - Green
                    # B04 (665 nm) - Red
                    # 注意：存储顺序需要调整为RGB
                    
                    blue = nc.variables['B02'][:].astype(np.float32)   # Blue
                    green = nc.variables['B03'][:].astype(np.float32)  # Green
                    red = nc.variables['B04'][:].astype(np.float32)    # Red
                    
                    # 堆叠为 (H, W, 3) RGB格式
                    image = np.stack([red, green, blue], axis=-1)
                    
                    # 读取标签
                    mask = nc.variables['Label'][:].astype(np.uint8)
                
                # 成功读取，跳出重试循环
                break
                
            except (RuntimeError, OSError) as e:
                if attempt < max_retries - 1:
                    # 如果不是最后一次尝试，等待后重试
                    time.sleep(retry_delay * (attempt + 1))  # 递增等待时间
                    continue
                else:
                    # 最后一次尝试失败，记录警告并返回空数据
                    warnings.warn(f"Failed to read {nc_path} after {max_retries} attempts: {str(e)}. Skipping this sample.")
                    # 返回全零数据作为fallback
                    image = np.zeros((512, 512, 3), dtype=np.float32)
                    mask = np.full((512, 512), 255, dtype=np.uint8)  # 全部标记为ignore
        
        # 归一化策略
        # NetCDF数据为反射率(0-1)，但通常很暗(均值~0.02)
        # 使用max_reflectance进行拉伸，将[0, max_reflectance]映射到[0, 1]
        max_ref = self.config['dataset']['normalization'].get('max_reflectance', 0.3)
        image = image / max_ref
        image = np.clip(image, 0.0, 1.0)  # 截断高光像素
        
        # 应用增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 标签重映射 (Label Remapping)
        # 原始: 0(Undef), 1(Clear), 2(Shadow), 3(Semi), 4(Cloud), 5(Miss)
        # 目标: 0(Clear), 1(Shadow), 2(Semi), 3(Cloud), 255(Ignore)
        
        if isinstance(mask, torch.Tensor):
            target = torch.full_like(mask, 255)
            target[mask == 1] = 0  # Clear
            target[mask == 2] = 1  # Shadow
            target[mask == 3] = 2  # Semi-transparent
            target[mask == 4] = 3  # Cloud
        else:
            target = np.full_like(mask, 255, dtype=np.uint8)
            target[mask == 1] = 0  # Clear
            target[mask == 2] = 1  # Shadow
            target[mask == 3] = 2  # Semi-transparent
            target[mask == 4] = 3  # Cloud
            target = torch.from_numpy(target)
        
        # --- 就在这里加上这一行防御性代码 ---
        # 强制将所有不在 0,1,2,3 范围内的异常像素全部设为 255 (忽略类)
        target[(target < 0) | (target > 3)] = 255
        
        return image, target.long()

def get_transforms(config, split='train'):
    """生成Albumentations增强流水线"""
    # 获取配置文件里的目标尺寸，通常是 512
    input_size = config['dataset'].get('input_size', 512) # 这里默认512，如果配置里没有就用512
    # 归一化参数
    mean = config['dataset']['normalization']['mean']
    std = config['dataset']['normalization']['std']

    if is_crop_mode(config):
        transforms = []
        if split == 'train':
            padding_fill = {'fill': 0, 'fill_mask': 255}
            if 'fill_mask' not in inspect.signature(A.PadIfNeeded).parameters:
                padding_fill = {'value': 0, 'mask_value': 255}
            transforms.extend([
                A.PadIfNeeded(
                    min_height=input_size, min_width=input_size,
                    border_mode=cv2.BORDER_CONSTANT, **padding_fill,
                ),
                A.RandomCrop(height=input_size, width=input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])
        transforms.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        return A.Compose(transforms)
    
    if split == 'train':
        affine_options = {}
        if (config['dataset'].get('type') == 'seo'
                and seo_supervision_options(config)['mode'] == 'minmax_agreement'):
            affine_options = {'border_mode': cv2.BORDER_CONSTANT,
                              **_nir_border_options(A.ShiftScaleRotate)}
        return A.Compose([
            A.Resize(input_size, input_size),
            # 几何增强 - 云影检测需要学习空间关系
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                              p=0.5, **affine_options),
            
            # 光度增强 - 模拟不同大气条件
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # 移除GaussNoise以避免OpenCV类型错误
            # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # 归一化与转换
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

def seo_supervision_options(config):
    """Normalize SEO label settings; omitted settings preserve legacy Max labels."""
    dataset = config['dataset']
    options = dataset.get('supervision', {})
    if not isinstance(options, dict):
        raise ValueError('dataset.supervision must be a configuration mapping')
    mode = options.get('mode', 'max')
    if mode not in {'max', 'minmax_agreement'}:
        raise ValueError('dataset.supervision.mode must be max or minmax_agreement')
    ignore_uncertainty = options.get('ignore_uncertainty', mode == 'minmax_agreement')
    if not isinstance(ignore_uncertainty, bool):
        raise ValueError('dataset.supervision.ignore_uncertainty must be a boolean')
    if mode == 'max' and ignore_uncertainty:
        raise ValueError('ignore_uncertainty requires minmax_agreement supervision')
    missing_policy = options.get('missing_policy', 'error')
    if missing_policy not in {'error', 'skip'}:
        raise ValueError('dataset.supervision.missing_policy must be error or skip')
    if mode == 'minmax_agreement' and config.get('model', {}).get('ignore_index', 255) != 255:
        raise ValueError('SEO Min/Max supervision requires model.ignore_index: 255')
    root = Path(dataset['root_dir'])

    def resolve(name, default):
        path = Path(options.get(name, default))
        return path if path.is_absolute() else root / path

    return {
        'mode': mode, 'ignore_uncertainty': ignore_uncertainty,
        'missing_policy': missing_policy,
        'min_root': resolve('min_root', 'shadows_min/shadows'),
        'max_root': resolve('max_root', 'shadows_max/shadow_max_compressed'),
    }


def _seo_supervision_paths(mask_path, options):
    mask = Path(mask_path)
    minimum = options['min_root'] / 'shadow_masks' / mask.parent.name / mask.name
    paths = [minimum]
    if options['ignore_uncertainty']:
        uncertainty = mask.name.removesuffix('_shadow.png') + '_uncertainty_mask.png'
        paths.append(options['min_root'] / 'uncertainty_masks' / mask.parent.name / uncertainty)
    return paths


def _filter_seo_supervision(selected, pairs, options, split):
    """Filter only after the existing AOI split and NIR selection, with a report."""
    required_directories = [options['min_root'] / 'shadow_masks']
    if options['ignore_uncertainty']:
        required_directories.append(options['min_root'] / 'uncertainty_masks')
    for directory in required_directories:
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing SEO Min directory: {directory}. '
                                    'Extract shadows_min.tar.gz into dataset.root_dir/shadows_min; '
                                    'keep the archive shadows/ subdirectory.')
    complete, missing = [], []
    for sample in selected:
        index = sample[0]
        absent = [path for path in _seo_supervision_paths(pairs[index][1], options)
                  if not path.is_file()]
        if absent:
            if options['missing_policy'] == 'error':
                raise FileNotFoundError(f'Missing SEO Min/Max supervision file: {absent[0]}. '
                                        'Extract shadows_min.tar.gz or explicitly set '
                                        'dataset.supervision.missing_policy: skip for incomplete pairs.')
            missing.append(pairs[index][2] + '/' + Path(pairs[index][0]).name.removesuffix('_rgb.png'))
        else:
            complete.append(sample)
    if not complete:
        raise ValueError(f'No complete Min/Max supervision samples remain in SEO {split} split')
    report = {'split': split, 'before': len(selected), 'after': len(complete),
              'missing': len(missing), 'missing_examples': missing[:5],
              'missing_policy': options['missing_policy']}
    return complete, report


class AISDDataset(Dataset):
    def __init__(self, img_paths, mask_paths, config, transform=None):
        if len(img_paths) != len(mask_paths):
            raise ValueError("AISD image and mask lists must have the same length")
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.config = config
        self.transform = transform
        self.supervision_options = (seo_supervision_options(config)
                                    if config['dataset'].get('type') == 'seo' else None)
        if self.supervision_options and self.supervision_options['mode'] == 'minmax_agreement':
            for mask_path in self.mask_paths:
                for path in _seo_supervision_paths(mask_path, self.supervision_options):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f'Missing SEO Min/Max supervision file: {path}. '
                            'Extract shadows_min.tar.gz into dataset.root_dir/shadows_min; '
                            'keep the archive shadows/ subdirectory.'
                        )
        
    def __len__(self):
        return len(self.img_paths)

    def _read_mask(self, idx, image_shape):
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Unable to read segmentation mask: {mask_path}')
        if tuple(image_shape) != mask.shape:
            raise ValueError(f'Image/mask spatial sizes differ: {image_shape} vs {mask.shape}: {mask_path}')
        if self.supervision_options is None:
            return cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
        target = (mask < 127).astype(np.uint8)
        if self.supervision_options['mode'] == 'minmax_agreement':
            for index, path in enumerate(_seo_supervision_paths(mask_path, self.supervision_options)):
                extra = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if extra is None:
                    raise ValueError(f'Unable to read SEO supervision mask: {path}')
                if extra.shape != mask.shape:
                    raise ValueError(f'SEO Min/Max/uncertainty spatial sizes differ: '
                                     f'{mask.shape} vs {extra.shape}: {path}')
                # On agreement pixels the Min and Max class labels are identical.
                # Official filtering uses Min-side uncertainty, with any nonzero invalid.
                target[(extra != mask) if index == 0 else (extra != 0)] = 255
        return target

    def __getitem__(self, idx):
        # 1. 读取影像 (AISD 是标准图像，用 cv2 读取)
        image = cv2.imread(self.img_paths[idx])
        if image is None:
            raise ValueError(f"Unable to read AISD image: {self.img_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 读取掩膜 (单通道灰度图)
        mask = self._read_mask(idx, image.shape[:2])
        
        # 4. 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        image_shape = tuple(image.shape[-2:]) if isinstance(image, torch.Tensor) else image.shape[:2]
        if image_shape != tuple(mask.shape):
            raise ValueError(f"Transformed image/mask spatial sizes differ: {image_shape} vs {tuple(mask.shape)}")
        return image, mask.long() # 确保掩膜是 long 类型以适配损失函数

def load_aligned_nir(nir_path, valid_path, rgb_shape, nir_options):
    """Read aligned raw DN as normalized float32 HxW and boolean HxW support."""
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError:
        rasterio = None

    raster_mask = None
    if rasterio is None:
        from PIL import Image
        with Image.open(nir_path) as source:
            nir = np.asarray(source, dtype=np.float32)
        if nir.ndim != 2:
            raise ValueError(f"Aligned NIR must have one band: {nir_path}")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NotGeoreferencedWarning)
            with rasterio.open(nir_path) as source:
                if source.count != 1:
                    raise ValueError(f"Aligned NIR must have one band: {nir_path}")
                nir = source.read(1, out_dtype='float32')
                raster_mask = source.read_masks(1)
    png = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
    if png is None:
        raise ValueError(f"Unable to read NIR validity mask: {valid_path}")
    expected = tuple(rgb_shape)
    if nir.shape != expected or png.shape != expected:
        raise ValueError(f"RGB/NIR/valid spatial sizes differ: {expected}, {nir.shape}, {png.shape}")
    # Reuse the byte mask and bound finite-value checks to small row blocks.
    np.greater(png, np.uint8(0), out=png)
    valid = png.view(np.bool_)
    for top in range(0, nir.shape[0], 128):
        block = slice(top, top + 128)
        np.logical_and(valid[block], np.isfinite(nir[block]), out=valid[block])
        if raster_mask is not None:
            np.logical_and(valid[block], raster_mask[block], out=valid[block])
    del raster_mask
    percentiles = nir_options.get('normalization', {}).get('percentiles', [1.0, 99.0])
    if (len(percentiles) != 2 or not np.isfinite(percentiles).all()
            or not 0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("dataset.nir.normalization.percentiles must satisfy 0 <= low < high <= 100")
    normalized = np.zeros(nir.shape, dtype=np.float32)
    if valid.any():
        # Percentiles may partition their private copy; keep source pixel order intact.
        percentile_values = nir[valid]
        low, high = np.percentile(percentile_values, percentiles, overwrite_input=True)
        del percentile_values
        if high > low:
            # Float64 percentile scalars otherwise promote whole-image temporaries.
            np.subtract(nir, np.float32(low), out=normalized)
            np.divide(normalized, np.float32(high - low), out=normalized)
            np.clip(normalized, np.float32(0), np.float32(1), out=normalized)
            np.multiply(normalized, np.float32(2), out=normalized)
            np.subtract(normalized, np.float32(1), out=normalized)
            normalized[~valid] = 0
    return normalized, valid


def _nir_border_options(transform_class):
    if 'fill_mask' in inspect.signature(transform_class).parameters:
        return {'fill': 0, 'fill_mask': 255}
    return {'value': 0, 'mask_value': 255}


class SEONIRTransforms:
    """Share geometry across four fields, then normalize only the RGB image."""
    def __init__(self, config, split='train'):
        self.split = split
        self.crop_mode = is_crop_mode(config)
        options = config['dataset'].get('nir', {})
        self.crop_retries = options.get('crop_retries', 4)
        self.min_valid_fraction = options.get('min_valid_fraction', 0.05)
        if (isinstance(self.crop_retries, bool) or not isinstance(self.crop_retries, int)
                or self.crop_retries < 0):
            raise ValueError("dataset.nir.crop_retries must be a nonnegative integer")
        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("dataset.nir.min_valid_fraction must be between zero and one")
        input_size = config['dataset'].get('input_size', 512)
        geometry = []
        if self.crop_mode:
            if split == 'train':
                geometry.extend([
                    A.PadIfNeeded(min_height=input_size, min_width=input_size,
                                  border_mode=cv2.BORDER_CONSTANT, **_nir_border_options(A.PadIfNeeded)),
                    A.RandomCrop(height=input_size, width=input_size),
                ])
        else:
            geometry.append(A.Resize(input_size, input_size))
        if split == 'train':
            geometry.extend([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
            if not self.crop_mode:
                geometry.append(A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5,
                    border_mode=cv2.BORDER_CONSTANT, **_nir_border_options(A.ShiftScaleRotate)))
        # Continuous support is interpolated with NIR; only fully supported pixels survive.
        self.geometry = A.Compose(geometry, additional_targets={'nir': 'image', 'nir_valid': 'image'})
        rgb = []
        if split == 'train':
            rgb.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        normalization = config['dataset']['normalization']
        rgb.extend([A.Normalize(mean=normalization['mean'], std=normalization['std']), ToTensorV2()])
        self.rgb = A.Compose(rgb)

    def __call__(self, *, image, mask, nir, nir_valid):
        if image.shape[:2] != mask.shape or nir.shape != mask.shape or nir_valid.shape != mask.shape:
            raise ValueError("Joint RGB/NIR/valid/label spatial sizes differ")
        attempts = 1 + self.crop_retries if self.crop_mode and self.split == 'train' and nir_valid.any() else 1
        best, best_fraction = None, -1.0
        for _ in range(attempts):
            transformed = self.geometry(image=image, mask=mask, nir=nir,
                                        nir_valid=nir_valid.astype(np.float32))
            support = transformed['nir_valid'] >= 1 - 1e-6
            fraction = float(support.mean())
            if fraction > best_fraction:
                best, best_fraction = transformed, fraction
            if fraction >= self.min_valid_fraction:
                break
        valid = best['nir_valid'] >= 1 - 1e-6
        normalized = np.asarray(best['nir'], dtype=np.float32).copy()
        valid &= np.isfinite(normalized)
        normalized[~valid] = 0
        return {
            'image': self.rgb(image=best['image'])['image'],
            'mask': torch.from_numpy(np.ascontiguousarray(best['mask'])).long(),
            'nir': torch.from_numpy(np.ascontiguousarray(normalized))[None],
            'nir_valid': torch.from_numpy(np.ascontiguousarray(valid))[None],
        }


def get_nir_transforms(config, split='train'):
    return SEONIRTransforms(config, split)


class SEONIRDataset(AISDDataset):
    def __init__(self, img_paths, mask_paths, nir_paths, valid_paths, config, split='train', transform=None):
        if len(img_paths) != len(nir_paths) or len(img_paths) != len(valid_paths):
            raise ValueError("SEO RGB/NIR/valid lists must have the same length")
        if transform is not None and not isinstance(transform, SEONIRTransforms):
            raise ValueError("SEO NIR requires get_nir_transforms; RGB-only external transforms are unsupported")
        super().__init__(img_paths, mask_paths, config, transform or get_nir_transforms(config, split))
        self.nir_paths = nir_paths
        self.valid_paths = valid_paths

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        if image is None:
            raise ValueError(f"Unable to read SEO image: {self.img_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._read_mask(idx, image.shape[:2])
        nir, valid = load_aligned_nir(self.nir_paths[idx], self.valid_paths[idx],
                                      image.shape[:2], self.config['dataset']['nir'])
        return self.transform(image=image, mask=mask, nir=nir, nir_valid=valid)


def _seo_nir_manifest(config):
    options = config['dataset']['nir']
    root = Path(options.get('root_dir', Path(config['dataset']['root_dir']) / 'nir_aligned_v1'))
    manifest = Path(options.get('manifest', 'registration_with_candidates.csv'))
    if not manifest.is_absolute():
        manifest = root / manifest
    records = {}
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'key', 'status', 'reference_grid', 'nir', 'valid'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"SEO NIR manifest requires columns: {sorted(required)}")
        for row in reader:
            if row['key'] in records:
                raise ValueError(f"Duplicate SEO NIR manifest key: {row['key']}")
            records[row['key']] = row
    allowed = {'original_rgb_ready', 'rgb_ready'}
    if options.get('include_candidates', False):
        allowed.add('candidate_rgb')
    eligible = {}
    for key, row in records.items():
        if row['reference_grid'] != 'RGB' or row['status'] not in allowed:
            continue
        paths = []
        for name in ('nir', 'valid'):
            if not row[name]:
                raise ValueError(f"Missing {name} path for eligible SEO NIR sample: {key}")
            path = Path(row[name])
            paths.append(str(path if path.is_absolute() else root / path))
        eligible[key] = tuple(paths)
    return eligible


def resolve_seo_nir_paths(config, image_path):
    """Resolve an RGB pair through the selected registration manifest."""
    image = Path(image_path)
    key = image.parent.name + '/' + image.name.removesuffix('_rgb.png')
    eligible = _seo_nir_manifest(config)
    if key not in eligible:
        raise ValueError(f"SEO image has no selected RGB-grid registration: {key}")
    nir_path, valid_path = eligible[key]
    if not Path(nir_path).is_file() or not Path(valid_path).is_file():
        raise ValueError(f"Missing eligible SEO NIR/valid file: {nir_path}, {valid_path}")
    return nir_path, valid_path


def _collect_aisd_pairs(split_dir):
    """Collect explicitly paired shadow/mask files from one fixed split."""
    image_dir = os.path.join(split_dir, 'shadow')
    mask_dir = os.path.join(split_dir, 'mask')
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise ValueError(f"AISD split requires shadow/ and mask/ folders: {split_dir}")

    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

    def index_files(directory):
        indexed = {}
        for path in sorted(glob(os.path.join(directory, '*'))):
            stem, extension = os.path.splitext(os.path.basename(path))
            if not os.path.isfile(path) or extension.lower() not in extensions:
                continue
            key = stem.lower()
            if key in indexed:
                raise ValueError(f"Duplicate AISD filename stem '{stem}' in {directory}")
            indexed[key] = path
        return indexed

    images = index_files(image_dir)
    masks = index_files(mask_dir)
    missing_masks = sorted(set(images) - set(masks))
    if missing_masks:
        raise ValueError(f"Missing AISD masks in {mask_dir}: {missing_masks[:5]}")
    if not images:
        raise ValueError(f"No AISD images found in {image_dir}")
    keys = sorted(images)
    return [images[key] for key in keys], [masks[key] for key in keys]


def _get_aisd_paths(config, split):
    if split not in {'train', 'val', 'test'}:
        raise ValueError(f"Invalid AISD split: {split}")
    dataset_config = config['dataset']
    split_mode = dataset_config.get('split_mode', 'random').lower()
    if split_mode not in {'random', 'predefined'}:
        raise ValueError("dataset.split_mode must be 'random' or 'predefined'")
    if split_mode == 'predefined' or split == 'test':
        directory_key = f'{split}_dir'
        if directory_key not in dataset_config:
            raise ValueError(f"AISD {split} split requires dataset.{directory_key}")
        split_dir = dataset_config[directory_key]
        if not os.path.isabs(split_dir):
            split_dir = os.path.join(dataset_config['root_dir'], split_dir)
        return _collect_aisd_pairs(split_dir)

    # Preserve the legacy random split, with a stable input ordering.
    all_files = sorted(glob(os.path.join(dataset_config['root_dir'], '**', '*.tif'), recursive=True))
    
    candidates_img = []
    candidates_mask = {}
    
    for f in all_files:
        if "mask" in f.lower() or "label" in f.lower(): 
            key = os.path.splitext(os.path.basename(f))[0].lower().replace("_mask", "").replace("_label", "")
            candidates_mask[key] = f  
        else:
            candidates_img.append(f) 
            
    img_list, mask_list = [], []
    for img_path in candidates_img:
        fname = os.path.splitext(os.path.basename(img_path))[0].lower()
        if fname in candidates_mask:
            img_list.append(img_path)
            mask_list.append(candidates_mask[fname])

    train_img, val_img, train_mask, val_mask = train_test_split(
        img_list, mask_list, test_size=0.1, random_state=42
    )
    if split == 'train':
        return train_img, train_mask
    return val_img, val_mask


def get_aisd_dataset(config, split='val', transform=None):
    """Build one AISD split without merging training, validation and test data."""
    img_paths, mask_paths = _get_aisd_paths(config, split)
    if transform is None:
        transform = get_transforms(config, split)
    return AISDDataset(img_paths, mask_paths, config, transform=transform)


def get_aisd_loaders(config):
    train_ds = get_aisd_dataset(config, 'train')
    val_ds = get_aisd_dataset(config, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, 
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
    val_batch_size = 1 if is_crop_mode(config) else config['training']['batch_size']
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, 
                            num_workers=config['num_workers'], pin_memory=config['pin_memory'])
                            
    return train_loader, val_loader


# SEO 数据集加载器。按JAX_100的地块编号分组划分，避免同一地块的不同视角同时进入训练和验证集。
def get_seo_dataset(config, split='val', transform=None):
    """Build the same grouped SEO split for training and standalone inference."""
    if split == 'test':
        raise ValueError("SEO has no independent test split configured; use 'val' explicitly")
    if split not in {'train', 'val'}:
        raise ValueError(f"Invalid SEO split: {split}")
    root = Path(config["dataset"]["root_dir"])
    supervision = seo_supervision_options(config)
    mask_root = supervision['max_root'] / 'shadow_masks'
    pairs = []
    for image in sorted((root / "pansharpened_he").rglob("*_rgb.png")):
        mask = mask_root / image.parent.name / image.name.replace("_rgb.png", "_shadow.png")
        if mask.is_file():
            pairs.append((str(image), str(mask), image.parent.name))
    if not pairs:
        raise ValueError("No paired SEO images and masks found")
    groups = [p[2] for p in pairs]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config["dataset"].get("val_ratio", 0.2),
        random_state=config["random_seed"])
    train_idx, val_idx = next(splitter.split(pairs, groups=groups))
    indices = train_idx if split == 'train' else val_idx
    nir_enabled = config['dataset'].get('nir', {}).get('enabled', False)
    if nir_enabled:
        eligible = _seo_nir_manifest(config)
        selected = []
        for index in indices:
            key = pairs[index][2] + '/' + Path(pairs[index][0]).name.removesuffix('_rgb.png')
            if key in eligible:
                selected.append((index, *eligible[key]))
        if not selected:
            raise ValueError(f"No RGB-aligned NIR samples remain in SEO {split} split")
        for _, nir_path, valid_path in selected:
            if not Path(nir_path).is_file() or not Path(valid_path).is_file():
                raise ValueError(f"Missing eligible SEO NIR/valid file: {nir_path}, {valid_path}")
    else:
        selected = [(index, None, None) for index in indices]
    nir_count = len(selected)
    report = None
    if supervision['mode'] == 'minmax_agreement':
        selected, report = _filter_seo_supervision(selected, pairs, supervision, split)
    if nir_enabled:
        dataset = SEONIRDataset(
            [pairs[index][0] for index, _, _ in selected],
            [pairs[index][1] for index, _, _ in selected],
            [nir_path for _, nir_path, _ in selected],
            [valid_path for _, _, valid_path in selected],
            config, split=split, transform=transform)
        dataset.nir_filter_report = {'split': split, 'before': len(indices), 'after': nir_count,
                                     'include_candidates': config['dataset']['nir'].get('include_candidates', False)}
    else:
        if transform is None:
            transform = get_transforms(config, split)
        dataset = AISDDataset(
            [pairs[index][0] for index, _, _ in selected],
            [pairs[index][1] for index, _, _ in selected], config, transform=transform)
    dataset.supervision_filter_report = report
    return dataset


def get_seo_loaders(config):
    train_ds = get_seo_dataset(config, 'train')
    val_ds = get_seo_dataset(config, 'val')
    kwargs = dict(batch_size=config["training"]["batch_size"],
                  num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    if config["num_workers"] > 0 and "prefetch_factor" in config:
        kwargs['prefetch_factor'] = config['prefetch_factor']
    val_kwargs = dict(kwargs)
    if is_crop_mode(config):
        val_kwargs['batch_size'] = 1
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **val_kwargs))

"""CPU data checks before starting the portable SEO training bundle.

Run from segformer/, for example /data/data_set/segformer. Relative config,
data (normally ../SEO), manifest, and checkpoint paths resolve from that cwd.
No model is built, no checkpoint is loaded, and only one image per split is read.
"""

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform


os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
RUN_HINT = ('Run from the segformer working directory, e.g. /data/data_set/segformer. '
            'Relative data paths such as ../SEO resolve from the working directory.')


def _portable_config_path(value, label):
    text = str(value)
    if PureWindowsPath(text).drive or '\\' in text:
        raise ValueError(f'{label} contains a Windows path; use portable forward-slash paths: {text}')


def check_manifest_paths(config):
    """Check selected rows without opening any NIR or RGB image."""
    dataset = config['dataset']
    _portable_config_path(dataset['root_dir'], 'dataset.root_dir')
    for name in ('min_root', 'max_root'):
        if name in dataset.get('supervision', {}):
            _portable_config_path(dataset['supervision'][name], 'dataset.supervision.' + name)
    options = dataset.get('nir', {})
    if not options.get('enabled', False):
        return None
    for name in ('root_dir', 'manifest'):
        if name in options:
            _portable_config_path(options[name], 'dataset.nir.' + name)
    root = Path(options.get('root_dir', Path(dataset['root_dir']) / 'nir_aligned_v1'))
    manifest = Path(options.get('manifest', 'registration_with_candidates.csv'))
    if not manifest.is_absolute():
        manifest = root / manifest
    allowed = {'original_rgb_ready', 'rgb_ready'}
    if options.get('include_candidates', False):
        allowed.add('candidate_rgb')
    seen, eligible = set(), 0
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'key', 'status', 'reference_grid', 'nir', 'valid'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'NIR manifest is missing columns: {sorted(required)}')
        for row in reader:
            if row['key'] in seen:
                raise ValueError('Duplicate NIR manifest key: ' + row['key'])
            seen.add(row['key'])
            if row['status'] not in allowed or row['reference_grid'] != 'RGB':
                continue
            for name in ('nir', 'valid'):
                text = row[name]
                relative = PurePosixPath(text)
                if (not text or text != text.strip() or PureWindowsPath(text).drive
                        or '\\' in text or relative.is_absolute() or '..' in relative.parts):
                    raise ValueError(f'NIR manifest {name} for {row["key"]} must be a relative POSIX '
                                     'path under dataset.nir.root_dir, e.g. AOI/image_nir.tif; '
                                     f'found {text!r}')
            eligible += 1
    return {'path': str(manifest.resolve()), 'rows': len(seen), 'eligible_rows': eligible,
            'row_paths_relative_to': str(root.resolve())}


def environment_report(torch, check_gpu):
    versions = {}
    for name in ('numpy', 'torchvision', 'segmentation-models-pytorch', 'timm',
                 'albumentations', 'scikit-learn', 'PyYAML', 'Pillow', 'rasterio',
                 'opencv-python-headless', 'opencv-python'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': str(torch.__version__), 'torch_cuda_build': torch.version.cuda,
              'packages': versions, 'gpu_checked': bool(check_gpu)}
    if check_gpu:
        report['cuda_available'] = bool(torch.cuda.is_available())
        if not report['cuda_available']:
            raise ValueError('--check-gpu requested, but torch.cuda.is_available() is false; '
                             f'PyTorch CUDA build: {torch.version.cuda}')
        report['devices'] = []
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            report['devices'].append({'index': index, 'name': device.name,
                                      'compute_capability': [device.major, device.minor],
                                      'total_memory_bytes': int(device.total_memory)})
    return report


def _sample_report(dataset, torch, nir_enabled):
    sample = dataset[0]
    if isinstance(sample, dict):
        image, target = sample['image'], sample['mask']
    else:
        image, target = sample
    if (not isinstance(image, torch.Tensor) or image.ndim != 3
            or image.shape[0] != 3 or image.dtype != torch.float32
            or not torch.isfinite(image).all()):
        raise ValueError('Sample RGB must be a finite float32 3xHxW tensor')
    if (not isinstance(target, torch.Tensor) or target.ndim != 2
            or tuple(target.shape) != tuple(image.shape[-2:]) or target.dtype != torch.long):
        raise ValueError('Sample label must be an aligned long HxW tensor')
    labels, counts = target.unique(return_counts=True)
    if not set(labels.tolist()) <= {0, 1, 255}:
        raise ValueError(f'SEO sample has labels outside 0/1/255: {labels.tolist()}')
    sources = {'image': str(dataset.img_paths[0]), 'max_mask': str(dataset.mask_paths[0])}
    supervision = getattr(dataset, 'supervision_options', None)
    if supervision and supervision['mode'] == 'minmax_agreement':
        maximum = Path(dataset.mask_paths[0])
        sources['min_mask'] = str(supervision['min_root'] / 'shadow_masks' / maximum.parent.name / maximum.name)
        if supervision['ignore_uncertainty']:
            filename = maximum.name.removesuffix('_shadow.png') + '_uncertainty_mask.png'
            sources['uncertainty_mask'] = str(supervision['min_root'] / 'uncertainty_masks'
                                              / maximum.parent.name / filename)
    report = {'index': 0, 'sources': sources, 'image_shape': list(image.shape),
              'target_shape': list(target.shape), 'rgb_finite': True,
              'label_counts': {str(label): int(count) for label, count in zip(labels.tolist(), counts.tolist())}}
    if nir_enabled:
        if not isinstance(sample, dict) or not {'nir', 'nir_valid'} <= set(sample):
            raise ValueError('NIR-enabled data must return nir and nir_valid tensors')
        nir, valid = sample['nir'], sample['nir_valid']
        expected = (1, *target.shape)
        if (tuple(nir.shape) != expected or tuple(valid.shape) != expected
                or nir.dtype != torch.float32 or valid.dtype != torch.bool):
            raise ValueError('Sample NIR/support must be aligned float32/bool 1xHxW tensors')
        if not torch.isfinite(nir).all() or not torch.all(nir[~valid] == 0):
            raise ValueError('Sample NIR must be finite with zero outside its validity mask')
        if valid.any() and (nir[valid].min() < -1.000002 or nir[valid].max() > 1.000002):
            raise ValueError('Sample normalized NIR is outside [-1, 1]')
        sources.update(nir=str(dataset.nir_paths[0]), nir_valid=str(dataset.valid_paths[0]))
        report.update(nir_shape=list(nir.shape), nir_finite=True,
                      nir_valid_fraction=float(valid.float().mean().item()))
    return report


def run_preflight(config_path='config_seo_cca_nir_server.yaml', checkpoint=None, check_gpu=False):
    """Build both splits from file lists and read exactly one sample from each."""
    import yaml
    import torch
    from dataset import get_nir_transforms, get_seo_dataset, get_transforms, seo_supervision_options

    config_path = Path(config_path)
    with config_path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get('dataset', {}).get('type', '').lower() != 'seo':
        raise ValueError('This preflight requires an SEO configuration')
    checkpoint_report = None
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint file does not exist: {path}')
        checkpoint_report = {'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
                             'loaded': False}
    report = {'ok': False, 'working_directory': str(Path.cwd()), 'run_hint': RUN_HINT,
              'config': str(config_path.resolve()), 'environment': environment_report(torch, check_gpu),
              'checkpoint': checkpoint_report, 'manifest': check_manifest_paths(config)}
    supervision = seo_supervision_options(config)
    report['supervision'] = {key: str(value) if isinstance(value, Path) else value
                             for key, value in supervision.items()}
    nir_enabled = config['dataset'].get('nir', {}).get('enabled', False)
    datasets, groups = {}, {}
    for split in ('train', 'val'):
        transform = get_nir_transforms(config, 'val') if nir_enabled else get_transforms(config, 'val')
        dataset = get_seo_dataset(config, split, transform=transform)
        if not len(dataset):
            raise ValueError(f'SEO {split} dataset is empty')
        datasets[split] = dataset
        groups[split] = {Path(path).parent.name for path in dataset.img_paths}
    overlap = groups['train'] & groups['val']
    if overlap:
        raise ValueError(f'SEO train/val AOIs overlap: {sorted(overlap)[:5]}')
    report['aoi_disjoint'] = True
    split_hashes = {}
    for split, dataset in datasets.items():
        keys = [Path(path).parent.name + '/' + Path(path).name.removesuffix('_rgb.png')
                for path in dataset.img_paths]
        split_hashes[split] = hashlib.sha256(('\n'.join(keys) + '\n').encode()).hexdigest()
    report['split_key_sha256'] = split_hashes
    bundle_path = Path(config['dataset']['root_dir']) / 'BUNDLE_INFO.json'
    report['bundle_info'] = None
    if bundle_path.is_file():
        with bundle_path.open(encoding='utf-8') as stream:
            bundle = json.load(stream)
        for split, dataset in datasets.items():
            if bundle.get('expected_counts', {}).get(split) != len(dataset):
                raise ValueError(f'BUNDLE_INFO.json {split} count differs: actual={len(dataset)}, '
                                 f'expected={bundle.get("expected_counts", {}).get(split)}')
            if bundle.get('split_key_sha256', {}).get(split) != split_hashes[split]:
                raise ValueError(f'BUNDLE_INFO.json {split} ordered-key SHA-256 differs')
        report['bundle_info'] = {'path': str(bundle_path.resolve()), 'validated': True}
    report['sample_transforms'] = 'validation transforms; no random training crops'
    report['splits'] = {}
    for split, dataset in datasets.items():
        report['splits'][split] = {
            'images': len(dataset), 'aois': sorted(groups[split]),
            'nir_filter_report': getattr(dataset, 'nir_filter_report', None),
            'supervision_filter_report': getattr(dataset, 'supervision_filter_report', None),
            'sample': _sample_report(dataset, torch, nir_enabled),
        }
    report['ok'] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=RUN_HINT)
    parser.add_argument('--config', default='config_seo_cca_nir_server.yaml')
    parser.add_argument('--checkpoint', help='Only check existence and size; never deserialize this file')
    parser.add_argument('--check-gpu', action='store_true', help='Query CUDA availability/device metadata only')
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.config, checkpoint=args.checkpoint, check_gpu=args.check_gpu)
    except Exception as exc:
        report = {'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                  'working_directory': str(Path.cwd()), 'run_hint': RUN_HINT}
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

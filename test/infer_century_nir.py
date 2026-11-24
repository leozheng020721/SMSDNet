"""Infer rectified BLUE/GREEN/RED/NIR scenes using the SEO NIR checkpoint.

Run with the existing rgb Conda environment (PyTorch + Rasterio). Inputs are
read only. Windowed reads and stripe-wise logit averaging bound host memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_factory import build_model

DEFAULT_INPUT = Path(r'E:\Dataset\21century\registration_output\rectified')
DEFAULT_CHECKPOINT = ROOT / 'checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_best.pth'
DEFAULT_OUTPUT = ROOT / 'test/results_century_nir_minmax'
DEFAULT_DISABLED_OUTPUT = ROOT / 'test/results_century_nir_disabled'
ALGORITHM = 'uniform_overlap_logits_streaming_v1'
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def tile_starts(length, tile_size, stride):
    if length < 1 or tile_size < 1 or not 0 < stride <= tile_size:
        raise ValueError('Require positive dimensions and 0 < stride <= tile_size')
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def percentile_from_histogram(histogram, percentiles):
    histogram = np.asarray(histogram)
    percentiles = np.asarray(percentiles, dtype=np.float64)
    if histogram.ndim != 1 or np.any(histogram < 0):
        raise ValueError('Histogram must be a nonnegative one-dimensional array')
    if not np.isfinite(percentiles).all() or np.any((percentiles < 0) | (percentiles > 100)):
        raise ValueError('Percentiles must be finite values in [0, 100]')
    cumulative = np.cumsum(histogram, dtype=np.int64)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total == 0:
        return np.zeros(percentiles.shape, dtype=np.float64)
    rank = (total - 1) * percentiles / 100.0
    lower, upper = np.floor(rank).astype(np.int64), np.ceil(rank).astype(np.int64)
    left = np.searchsorted(cumulative, lower + 1).astype(np.float64)
    right = np.searchsorted(cumulative, upper + 1).astype(np.float64)
    return left + (right - left) * (rank - lower)


def validate_source(source):
    names = tuple((name or '').strip().upper() for name in source.descriptions)
    if source.count != 4 or source.dtypes != ('uint16',) * 4:
        raise ValueError(f'Expected four uint16 bands: {source.name}')
    if names != ('BLUE', 'GREEN', 'RED', 'NIR'):
        raise ValueError(f'Unconfirmed band order {names}: {source.name}')


def scene_validity(source, data, window):
    valid = source.dataset_mask(window=window, boundless=True) > 0
    return valid & np.any(data != 0, axis=0)


def scene_statistics(source, percentiles):
    """Exact image-wide uint16 NIR percentiles, excluding invalid black borders."""
    validate_source(source)
    histogram = np.zeros(65536, dtype=np.int64)
    valid_pixels = 0
    for _, window in source.block_windows(1):
        data = source.read(window=window)
        valid = scene_validity(source, data, window)
        valid_pixels += int(np.count_nonzero(valid))
        histogram += np.bincount(data[3][valid], minlength=65536)
    low, high = percentile_from_histogram(histogram, percentiles)
    return {'valid_pixels': valid_pixels, 'nir_low': float(low), 'nir_high': float(high),
            'nir_percentiles': list(percentiles),
            'nir_percentile_algorithm': 'exact_valid_uint16_histogram_linear'}


def prepare_tile(source, x, y, tile_size, mean, std, dn_max, nir_low, nir_high):
    window = Window(x, y, tile_size, tile_size)
    data = source.read(window=window, boundless=True, fill_value=0)
    valid = scene_validity(source, data, window)
    rgb = data[[2, 1, 0]].astype(np.float32)
    rgb /= np.float32(dn_max)
    np.clip(rgb, 0, 1, out=rgb)
    rgb -= np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    rgb /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    rgb[:, ~valid] = 0
    nir = np.zeros((1, tile_size, tile_size), dtype=np.float32)
    if nir_high > nir_low:
        np.subtract(data[3], np.float32(nir_low), out=nir[0], casting='unsafe')
        nir /= np.float32(nir_high - nir_low)
        np.clip(nir, 0, 1, out=nir)
        nir *= np.float32(2)
        nir -= np.float32(1)
        nir[:, ~valid] = 0
    return rgb, nir, valid[None], valid


def stream_predict(model, source, device, mean, std, nir_low, nir_high,
                   mask_writer, valid_writer, probability_writer=None, *,
                   tile_size=512, stride=384, batch_size=2, dn_max=4095.0,
                   amp_enabled=True, threshold=0.5, disable_nir=False):
    device = torch.device(device)
    rows = tile_starts(source.height, tile_size, stride)
    columns = tile_starts(source.width, tile_size, stride)
    carry_sum = carry_weight = None
    valid_pixels = shadow_pixels = completed = 0
    for row_index, top in enumerate(rows):
        height = min(tile_size, source.height - top)
        logit_sum = np.zeros((2, height, source.width), dtype=np.float32)
        coverage = np.zeros((height, source.width), dtype=np.float32)
        if carry_sum is not None:
            logit_sum[:, :carry_sum.shape[1]] = carry_sum
            coverage[:carry_weight.shape[0]] = carry_weight
        for offset in range(0, len(columns), batch_size):
            positions = columns[offset:offset + batch_size]
            tiles = [prepare_tile(source, left, top, tile_size, mean, std, dn_max,
                                  nir_low, nir_high) for left in positions]
            images = torch.from_numpy(np.stack([tile[0] for tile in tiles])).to(device)
            nir = torch.from_numpy(np.stack([tile[1] for tile in tiles])).to(device)
            nir_valid = torch.from_numpy(np.stack([tile[2] for tile in tiles])).to(device)
            if disable_nir:
                # Disable NIR injection, keeping tiles[index][3] as scene support.
                nir_valid.zero_()
                nir.zero_()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=bool(amp_enabled and device.type == 'cuda'),
            ):
                logits = model(images, nir=nir, nir_valid=nir_valid)
            expected = (len(positions), 2, tile_size, tile_size)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                raise ValueError(f'Model must return main logits with shape {expected}')
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Non-finite model logits; refusing to save a prediction')
            for index, left in enumerate(positions):
                width = min(tile_size, source.width - left)
                support = tiles[index][3][:height, :width]
                logit_sum[:, :, left:left + width] += values[index, :, :height, :width] * support
                coverage[:, left:left + width] += support
            completed += len(positions)
        flush_height = rows[row_index + 1] - top if row_index + 1 < len(rows) else height
        weight = coverage[:flush_height]
        valid = weight > 0
        average = np.zeros((2, flush_height, source.width), dtype=np.float32)
        np.divide(logit_sum[:, :flush_height], weight[None], out=average, where=valid[None])
        probability = 1.0 / (1.0 + np.exp(np.clip(average[0] - average[1], -80, 80)))
        shadow = ((average[1] > average[0]) if threshold == 0.5
                  else probability > threshold) & valid
        binary = shadow.astype(np.uint8) * 255
        valid_u8 = valid.astype(np.uint8) * 255
        window = Window(0, top, source.width, flush_height)
        mask_writer.write(binary, 1, window=window)
        valid_writer.write(valid_u8, 1, window=window)
        if probability_writer is not None:
            probability[~valid] = -1
            probability_writer.write(probability.astype(np.float32), 1, window=window)
        valid_pixels += int(np.count_nonzero(valid))
        shadow_pixels += int(np.count_nonzero(shadow))
        carry_sum = logit_sum[:, flush_height:].copy()
        carry_weight = coverage[flush_height:].copy()
        print(f'  tiles {completed}/{len(rows) * len(columns)}', flush=True)
    return {'tile_count': completed, 'valid_pixels': valid_pixels,
            'shadow_pixels': shadow_pixels,
            'shadow_fraction_of_valid': shadow_pixels / valid_pixels if valid_pixels else 0.0}


def output_profile(source, probability=False, compression='none'):
    profile = dict(driver='GTiff', width=source.width, height=source.height, count=1,
                   dtype='float32' if probability else 'uint8', crs=source.crs,
                   transform=source.transform, nodata=-1.0 if probability else None,
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    if compression == 'deflate':
        profile.update(compress='deflate', predictor=3 if probability else 2)
    return profile


def make_preview(source_path, mask_path, valid_path, destination, dn_max, disable_nir=False):
    with rasterio.open(source_path) as source, rasterio.open(mask_path) as masks, \
            rasterio.open(valid_path) as validity:
        factor = min(1.0, 1200 / max(source.height, source.width))
        height, width = max(1, round(source.height * factor)), max(1, round(source.width * factor))
        rgb = source.read([3, 2, 1], out_shape=(3, height, width),
                          resampling=Resampling.bilinear).astype(np.float32)
        valid = validity.read(1, out_shape=(height, width), resampling=Resampling.nearest) > 0
        shadow = masks.read(1, out_shape=(height, width), resampling=Resampling.nearest) > 0
    image = np.moveaxis((np.clip(rgb / dn_max, 0, 1) * 255).astype(np.uint8), 0, -1)
    image[~valid] = 0
    overlay = image.copy()
    overlay[shadow & valid] = (0.45 * overlay[shadow & valid]
                               + 0.55 * np.array([255, 40, 20])).astype(np.uint8)
    result = Image.new('RGB', (width * 2, height + 28), '#202020')
    result.paste(Image.fromarray(image), (0, 28))
    result.paste(Image.fromarray(overlay), (width, 28))
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), f'RGB input (DN / {dn_max:g})', fill='white')
    caption = 'NIR disabled' if disable_nir else 'NIR model'
    draw.text((width + 8, 7), f'{caption}: predicted shadow in red', fill='white')
    result.save(destination, quality=92)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' not in checkpoint:
        raise ValueError('A checkpoint with its training config is required')
    config = copy.deepcopy(checkpoint['config'])
    if not (config['model'].get('multicolor', {}).get('nir', {}).get('enabled', False)
            and config['dataset'].get('nir', {}).get('enabled', False)):
        raise ValueError('Checkpoint must contain the enabled NIR model and input branches')
    if config['dataset']['num_classes'] != 2 or config['dataset']['in_channels'] != 3:
        raise ValueError('Expected a binary RGB + separate NIR checkpoint')
    config['model']['encoder_weights'] = None
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    metadata = {'path': str(checkpoint_path), 'epoch_zero_based': checkpoint.get('epoch'),
                'stored_seo_validation_ber': checkpoint.get('ber'),
                'nir_enabled': True, 'config': config}
    del checkpoint
    model.to(device).eval()
    return model, config, metadata


def discover_inputs(input_root, groups, regions=None):
    images = []
    for group in groups:
        folder = input_root / str(group)
        if regions:
            folders = []
            for region in dict.fromkeys(regions):
                if Path(region).name != region or not region.startswith('region_'):
                    raise ValueError(f'Expected an exact region folder name: {region}')
                folders.append(folder / region)
        else:
            folders = sorted(folder.glob('region_*'))
        if not folders:
            raise FileNotFoundError(f'No rectified region folders in {folder}')
        for region_folder in folders:
            paths = sorted(region_folder.glob('image_[12]_rectified.tif'))
            if not paths:
                raise FileNotFoundError(f'No rectified TIFF pairs in {region_folder}')
            images.extend(paths)
    return images


def signatures_match(existing, current):
    if not isinstance(existing, dict):
        return False
    # Results created before the NIR switch always used real NIR.
    saved = dict(existing)
    saved.setdefault('nir_mode', 'real')
    return saved == current


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path,
                        help='Output folder; --disable-nir uses a separate default folder')
    parser.add_argument('--groups', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--regions', nargs='+',
                        help='Exact region folder names, e.g. region_01; default: all regions')
    parser.add_argument('--disable-nir', action='store_true',
                        help='Set NIR validity false without changing RGB or scene validity')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dn-max', type=float, default=4095.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--save-probability', action='store_true')
    parser.add_argument('--compression', choices=['none', 'deflate'], default='none')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if (args.tile_size <= 0 or args.tile_size % 32 or not 0 < args.stride <= args.tile_size
            or args.batch_size < 1 or not np.isfinite(args.dn_max) or args.dn_max <= 0
            or not 0 <= args.threshold <= 1 or (args.limit is not None and args.limit < 1)):
        parser.error('Invalid tile/stride/batch/DN/threshold/limit setting')
    if args.output is None:
        args.output = DEFAULT_DISABLED_OUTPUT if args.disable_nir else DEFAULT_OUTPUT
    args.input_root, args.output = args.input_root.resolve(), args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output == args.input_root or args.input_root in args.output.parents:
        parser.error('Keep outputs outside the read-only rectified source tree')
    sources = discover_inputs(args.input_root, args.groups, args.regions)
    if args.limit:
        sources = sources[:args.limit]
    for path in sources:
        with rasterio.open(path) as source:
            validate_source(source)
    requested = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if requested == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device = torch.device(requested)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    model, config, checkpoint_metadata = load_model(args.checkpoint, device)
    checkpoint_metadata['sha256'] = file_sha256(args.checkpoint)
    normalization = config['dataset']['normalization']
    mean, std = normalization['mean'], normalization['std']
    percentiles = config['dataset']['nir'].get('normalization', {}).get('percentiles', [1, 99])
    amp_enabled = device.type == 'cuda' and not args.fp32
    settings = dict(algorithm=ALGORITHM, checkpoint_sha256=checkpoint_metadata['sha256'],
                    rgb_bands=[3, 2, 1], nir_band=4, rgb_mapping='clip(DN / dn_max, 0, 1)',
                    dn_max=args.dn_max, mean=mean, std=std, nir_percentiles=percentiles,
                    tile_size=args.tile_size, stride=args.stride, batch_size=args.batch_size,
                    threshold=args.threshold, amp_enabled=amp_enabled,
                    save_probability=args.save_probability, compression=args.compression,
                    device=str(device), nir_mode='disabled' if args.disable_nir else 'real')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    print(f'Device: {device}; FP16 AMP: {amp_enabled}; NIR: {settings["nir_mode"]}; '
          f'images: {len(sources)}', flush=True)
    for number, path in enumerate(sources, 1):
        relative = path.relative_to(args.input_root)
        folder = args.output / relative.parent
        folder.mkdir(parents=True, exist_ok=True)
        stem = path.stem.removesuffix('_rectified')
        outputs = {name: folder / f'{stem}_{suffix}' for name, suffix in {
            'mask': 'shadow_mask.tif', 'valid': 'valid_mask.tif',
            'preview': 'preview.jpg', 'metadata': 'prediction.json',
            **({'probability': 'shadow_probability.tif'} if args.save_probability else {}),
        }.items()}
        source_stat = path.stat()
        signature = dict(settings, source=str(path), source_bytes=source_stat.st_size,
                         source_mtime_ns=source_stat.st_mtime_ns)
        if outputs['metadata'].is_file() and not args.overwrite:
            existing = json.loads(outputs['metadata'].read_text(encoding='utf-8'))
            if signatures_match(existing.get('signature'), signature) and all(p.is_file() for p in outputs.values()):
                print(f'[{number}/{len(sources)}] Existing verified signature: {relative}', flush=True)
                records.append(existing)
                continue
        if any(p.exists() for p in outputs.values()) and not args.overwrite:
            raise FileExistsError(f'Incomplete or different existing result: {folder}; use --overwrite explicitly')
        print(f'[{number}/{len(sources)}] {relative}', flush=True)
        started = time.monotonic()
        temporary = {key: value.with_name(value.stem + '.partial' + value.suffix)
                     for key, value in outputs.items() if key in ('mask', 'valid', 'probability')}
        from contextlib import ExitStack
        with rasterio.Env(GDAL_CACHEMAX=128 * 1024 ** 2), ExitStack() as stack:
            source = stack.enter_context(rasterio.open(path))
            statistics = scene_statistics(source, percentiles)
            profile = output_profile(source, compression=args.compression)
            mask_writer = stack.enter_context(rasterio.open(temporary['mask'], 'w', **profile))
            valid_writer = stack.enter_context(rasterio.open(temporary['valid'], 'w', **profile))
            probability_writer = (stack.enter_context(rasterio.open(
                temporary['probability'], 'w', **output_profile(
                    source, probability=True, compression=args.compression)))
                if 'probability' in temporary else None)
            result = stream_predict(
                model, source, device, mean, std, statistics['nir_low'], statistics['nir_high'],
                mask_writer, valid_writer, probability_writer,
                tile_size=args.tile_size, stride=args.stride, batch_size=args.batch_size,
                dn_max=args.dn_max, amp_enabled=amp_enabled, threshold=args.threshold,
                disable_nir=args.disable_nir,
            )
            if result['valid_pixels'] != statistics['valid_pixels']:
                raise RuntimeError('Streaming output validity differs from source validity')
            dimensions = dict(width=source.width, height=source.height, bands=list(source.descriptions),
                              crs=str(source.crs), transform=list(source.transform))
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError(f'Input changed during inference: {path}')
        for key, temp in temporary.items():
            temp.replace(outputs[key])
        make_preview(path, outputs['mask'], outputs['valid'], outputs['preview'], args.dn_max,
                     disable_nir=args.disable_nir)
        record = dict(signature=signature, checkpoint=checkpoint_metadata, dimensions=dimensions,
                      statistics=statistics, result=result,
                      outputs={key: str(value) for key, value in outputs.items()},
                      elapsed_seconds=time.monotonic() - started,
                      nir_inference_mode=settings['nir_mode'],
                      mask_encoding={'non_shadow': 0, 'shadow': 255, 'invalid': 'use separate valid mask'},
                      probability_encoding='float32 [0,1]; invalid=-1',
                      domain_note='SEO pansharpened_he training; Century RGB fixed DN mapping; cross-domain predictions')
        outputs['metadata'].write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        records.append(record)
        print(f'  complete: shadow={result["shadow_fraction_of_valid"]:.2%}; '
              f'{record["elapsed_seconds"]:.1f}s', flush=True)
    summary = dict(checkpoint=checkpoint_metadata, settings=settings,
                   image_count=len(records), groups=args.groups, regions=args.regions, results=records)
    (args.output / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Finished {len(records)} images: {args.output}', flush=True)


if __name__ == '__main__':
    main()

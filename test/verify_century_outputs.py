"""Independently audit completed Century NIR outputs using bounded block reads."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image
import rasterio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(r'E:\Dataset\21century\registration_output\rectified')
DEFAULT_OUTPUT = ROOT / 'test/results_century_nir_minmax'
EXPECTED_COUNTS = {'1': 10, '4': 12}
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_paths(source, input_root, output_root):
    relative = source.relative_to(input_root)
    folder = output_root / relative.parent
    stem = source.name.removesuffix('_rectified.tif')
    return {
        'mask': folder / f'{stem}_shadow_mask.tif',
        'valid': folder / f'{stem}_valid_mask.tif',
        'probability': folder / f'{stem}_shadow_probability.tif',
        'preview': folder / f'{stem}_preview.jpg',
        'metadata': folder / f'{stem}_prediction.json',
    }


def verify_one(source_path, input_root, output_root):
    paths = output_paths(source_path, input_root, output_root)
    for path in paths.values():
        require(path.is_file(), f'Missing output: {path}')
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    signature = metadata['signature']
    original_stat = source_path.stat()
    require(original_stat.st_size == signature['source_bytes'], 'Source byte size changed')
    require(original_stat.st_mtime_ns == signature['source_mtime_ns'], 'Source modification time changed')
    require(Path(signature['source']).resolve() == source_path.resolve(), 'Unexpected metadata source')
    require(signature['rgb_bands'] == [3, 2, 1] and signature['nir_band'] == 4,
            'Unexpected RGB or NIR input bands')
    require(metadata['checkpoint']['nir_enabled'] is True, 'Checkpoint is not NIR enabled')
    require(metadata['checkpoint']['config']['dataset']['supervision']['mode'] == 'minmax_agreement',
            'Checkpoint does not record Min/Max supervision')
    require(signature['checkpoint_sha256'] == metadata['checkpoint']['sha256'],
            'Conflicting checkpoint hashes')
    threshold = float(signature['threshold'])
    require(np.isfinite(threshold) and 0 <= threshold <= 1, 'Invalid probability threshold')
    require(signature['save_probability'] is True, 'Probability output was not requested')
    for key, path in paths.items():
        require(Path(metadata['outputs'][key]).resolve() == path.resolve(),
                f'Unexpected metadata output path: {key}')
    with Image.open(paths['preview']) as preview:
        require(preview.width > 0 and preview.height > 0, 'Empty preview')
        preview.verify()

    valid_count = shadow_count = checked_pixels = ambiguous_threshold_pixels = 0
    probability_min, probability_max = 1.0, 0.0
    mask_values, valid_values = set(), set()
    with rasterio.Env(GDAL_CACHEMAX=128 * 1024 ** 2), ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        names = tuple((name or '').strip().upper() for name in source.descriptions)
        require(source.count == 4 and source.dtypes == ('uint16',) * 4,
                'Source is not four-band uint16')
        require(names == ('BLUE', 'GREEN', 'RED', 'NIR'), 'Source band names do not confirm NIR')
        require(metadata['dimensions']['width'] == source.width
                and metadata['dimensions']['height'] == source.height, 'Incorrect metadata dimensions')
        datasets = {}
        for key, dtype in (('mask', 'uint8'), ('valid', 'uint8'), ('probability', 'float32')):
            dataset = stack.enter_context(rasterio.open(paths[key]))
            require((dataset.width, dataset.height, dataset.count, dataset.dtypes)
                    == (source.width, source.height, 1, (dtype,)), f'Incorrect {key} raster structure')
            require(dataset.transform == source.transform and dataset.crs == source.crs,
                    f'Incorrect {key} raster coordinate grid')
            datasets[key] = dataset
        require(datasets['probability'].nodata == -1, 'Probability nodata must be -1')
        for _, window in source.block_windows(1):
            data = source.read(window=window)
            expected_valid = (source.dataset_mask(window=window) > 0) & np.any(data != 0, axis=0)
            binary = datasets['mask'].read(1, window=window)
            valid = datasets['valid'].read(1, window=window)
            probability = datasets['probability'].read(1, window=window)
            mask_values.update(int(value) for value in np.unique(binary))
            valid_values.update(int(value) for value in np.unique(valid))
            require(mask_values.issubset({0, 255}), f'Unexpected mask values at {window}')
            require(valid_values.issubset({0, 255}), f'Unexpected valid-mask values at {window}')
            require(np.array_equal(valid > 0, expected_valid), f'Valid mask differs from source at {window}')
            require(np.isfinite(probability).all(), f'Non-finite probability at {window}')
            require(np.all(probability[~expected_valid] == -1), f'Invalid probability is not -1 at {window}')
            values = probability[expected_valid]
            if values.size:
                local_min, local_max = float(values.min()), float(values.max())
                require(local_min >= 0 and local_max <= 1, f'Probability out of range at {window}')
                probability_min = min(probability_min, local_min)
                probability_max = max(probability_max, local_max)
            expected_shadow = expected_valid & (probability > threshold)
            # Stored float32 probabilities can round to the threshold even
            # when the original logits lie just above or below the tie.
            ambiguous = expected_valid & (np.abs(probability - threshold) <= 1e-7)
            mismatch = (binary > 0) != expected_shadow
            require(not np.any(mismatch & ~ambiguous), f'Probability/mask threshold mismatch at {window}')
            ambiguous_threshold_pixels += int(np.count_nonzero(ambiguous))
            valid_count += int(np.count_nonzero(expected_valid))
            shadow_count += int(np.count_nonzero(binary))
            checked_pixels += int(binary.size)
        require(checked_pixels == source.width * source.height, 'Source blocks do not cover the whole image')
    require(valid_count == metadata['result']['valid_pixels'], 'Incorrect saved valid-pixel count')
    require(valid_count == metadata['statistics']['valid_pixels'], 'Incorrect normalization valid-pixel count')
    require(shadow_count == metadata['result']['shadow_pixels'], 'Incorrect saved shadow-pixel count')
    expected_fraction = shadow_count / valid_count if valid_count else 0.0
    saved_fraction = metadata['result']['shadow_fraction_of_valid']
    require((expected_fraction is None and saved_fraction is None)
            or (expected_fraction is not None and saved_fraction is not None
                and np.isclose(expected_fraction, saved_fraction, rtol=0, atol=1e-12)),
            'Incorrect shadow fraction')
    final_stat = source_path.stat()
    require((final_stat.st_size, final_stat.st_mtime_ns)
            == (original_stat.st_size, original_stat.st_mtime_ns), 'Source changed during audit')
    return {
        'key': str(source_path.relative_to(input_root)),
        'passed': True,
        'checked_pixels': checked_pixels,
        'valid_pixels': valid_count,
        'shadow_pixels': shadow_count,
        'shadow_fraction_of_valid': expected_fraction,
        'ambiguous_threshold_pixels': ambiguous_threshold_pixels,
        'mask_values': sorted(mask_values),
        'valid_values': sorted(valid_values),
        'valid_probability_range': [probability_min, probability_max] if valid_count else None,
        'checkpoint_sha256': signature['checkpoint_sha256'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = []
    counts = {}
    errors = []
    for group, expected in EXPECTED_COUNTS.items():
        files = sorted((args.input / group).glob('region_*/*_rectified.tif'))
        counts[group] = len(files)
        if len(files) != expected:
            errors.append(f'Group {group}: found {len(files)} sources, expected {expected}')
        sources.extend(files)
    expected_metadata = {output_paths(path, args.input, args.output)['metadata'].resolve() for path in sources}
    actual_metadata = {path.resolve() for group in EXPECTED_COUNTS
                       for path in (args.output / group).glob('region_*/image_*_prediction.json')}
    if expected_metadata != actual_metadata:
        errors.append(f'Metadata key set mismatch: missing={len(expected_metadata - actual_metadata)}, '
                      f'unexpected={len(actual_metadata - expected_metadata)}')
    results = []
    for number, path in enumerate(sources, 1):
        print(f'[{number}/{len(sources)}] {path.relative_to(args.input)}', flush=True)
        try:
            results.append(verify_one(path, args.input, args.output))
        except (OSError, ValueError, KeyError, TypeError, rasterio.errors.RasterioError) as error:
            results.append({'key': str(path.relative_to(args.input)), 'passed': False, 'error': str(error)})
    hashes = {row['checkpoint_sha256'] for row in results if row['passed']}
    if len(hashes) > 1:
        errors.append('Output records use different checkpoint hashes')
    passed = not errors and len(results) == 22 and all(row['passed'] for row in results)
    report = {'passed': passed, 'input': str(args.input.resolve()), 'output': str(args.output.resolve()),
              'expected_group_counts': EXPECTED_COUNTS, 'source_group_counts': counts,
              'image_count': len(results), 'errors': errors, 'results': results,
              'verification': 'independent windowed raster/source/metadata/probability audit; no model execution'}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'verification_report.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Audit passed={passed}: {destination}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

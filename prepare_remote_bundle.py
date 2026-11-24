"""Build portable SEO data and training-code tar archives without copying source data."""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import tarfile
import time

import yaml


CODE_FILES = [
    'train.py', 'inference.py', 'dataset.py', 'model_factory.py', 'model_multicolor.py',
    'model.py', 'modela.py', 'color_spaces.py', 'cross_scale_attention.py', 'losses.py',
    'lr_schedule.py', 'mixup.py', 'multimodal.py', 'patch_inference.py', 'metrics.py',
    'utils.py', 'aisd_inference.py', 'preflight_remote.py', 'prepare_remote_bundle.py',
    'requirements-server.txt', 'requirements.txt', 'config_seo_cca_nir_server.yaml',
    'config_seo_cca_nir.yaml', 'SEO_NIR_TRAINING_GUIDE.md', 'REMOTE_TRAINING.md',
    'test_metrics.py', 'test_losses.py', 'test_losses_nir.py', 'test_mixup.py',
    'test_lr_schedule.py', 'test_resume_memory.py', 'test_multicolor.py', 'test_model_nir.py',
    'test_cross_scale_attention.py', 'test_dataset_nir.py', 'test_seo_supervision.py',
    'test_nir_training_flow.py', 'test_patch_pipeline.py', 'test_training_flow.py',
    'test_inference_native.py', 'test_preflight_remote.py', 'test_prepare_remote_bundle.py',
    'test_mixed_precision.py',
    'checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth',
    'checkpoints/seo_segformer_native512_aux_cca_nir_mixup_history.csv',
]


class DigestWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data):
        count = self.stream.write(data)
        self.digest.update(memoryview(data)[:count])
        self.bytes_written += count
        return count

    def flush(self):
        self.stream.flush()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def _key(image):
    return image.parent.name + '/' + image.name.removesuffix('_rgb.png')


def _keys_digest(keys):
    return hashlib.sha256(('\n'.join(keys) + '\n').encode('utf-8')).hexdigest()


def data_plan(config):
    """Keep the full pre-NIR split base and all strictly accepted RGB-grid NIR."""
    from sklearn.model_selection import GroupShuffleSplit

    options = config['dataset']
    root = Path(options['root_dir']).resolve()
    supervision = options.get('supervision', {})

    def source_root(name, default):
        path = Path(supervision.get(name, default))
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    min_root = source_root('min_root', 'shadows_min/shadows')
    max_root = source_root('max_root', 'shadows_max/shadow_max_compressed')
    nir_root = Path(options['nir'].get('root_dir', root / 'nir_aligned_v1')).resolve()
    manifest = Path(options['nir'].get('manifest', 'registration_with_candidates.csv'))
    if not manifest.is_absolute():
        manifest = nir_root / manifest
    if options.get('nir', {}).get('include_candidates', False):
        raise ValueError('This bundle preserves strict RGB NIR only; include_candidates must be false')
    if supervision.get('mode') != 'minmax_agreement':
        raise ValueError('The remote bundle requires the current Min/Max agreement configuration')
    if not options['nir'].get('enabled', False):
        raise ValueError('The remote bundle requires dataset.nir.enabled: true')

    entries = {}
    components = {}
    relative_sets = {}

    def add(path, component):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Expected a regular source file: {path}')
        relative = path.relative_to(root).as_posix()
        archive_name = 'SEO/' + relative
        if archive_name in entries:
            return
        stat = path.stat()
        entries[archive_name] = (path, stat.st_size, stat.st_mtime_ns)
        details = components.setdefault(component, {'files': 0, 'bytes': 0})
        details['files'] += 1
        details['bytes'] += stat.st_size
        relative_sets.setdefault(component, set()).add(relative)

    sources = [
        ('rgb', root / 'pansharpened_he', '*_rgb.png'),
        ('max_shadow', max_root / 'shadow_masks', '*_shadow.png'),
        ('max_uncertainty', max_root / 'uncertainty_masks', '*_uncertainty_mask.png'),
        ('min_shadow', min_root / 'shadow_masks', '*_shadow.png'),
        ('min_uncertainty', min_root / 'uncertainty_masks', '*_uncertainty_mask.png'),
    ]
    for component, directory, pattern in sources:
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing required source directory: {directory}')
        for path in sorted(directory.rglob(pattern)):
            add(path, component)
        print(f'Inventory {component}: {components.get(component, {})}', flush=True)

    manifest_bytes = manifest.read_bytes()
    with io.StringIO(manifest_bytes.decode('utf-8-sig'), newline='') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not {'key', 'status', 'reference_grid', 'nir', 'valid'} <= set(fields or []):
            raise ValueError('NIR source manifest does not have the required columns')
        records, seen = [], set()
        for row in reader:
            if row['key'] in seen:
                raise ValueError(f'Duplicate NIR key: {row["key"]}')
            seen.add(row['key'])
            if row['reference_grid'] != 'RGB' or row['status'] not in {'original_rgb_ready', 'rgb_ready'}:
                continue
            for name in ('nir', 'valid'):
                path = Path(row[name])
                if not path.is_absolute():
                    path = nir_root / path
                relative = path.relative_to(nir_root)
                if '..' in relative.parts:
                    raise ValueError(f'NIR file escapes its root: {path}')
                row[name] = relative.as_posix()
                add(path, 'nir' if name == 'nir' else 'nir_valid')
            records.append(row)
    portable = io.StringIO(newline='')
    writer = csv.DictWriter(portable, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    portable_name = 'SEO/' + nir_root.relative_to(root).as_posix() + '/' + manifest.name
    generated = {portable_name: portable.getvalue().encode('utf-8')}

    # Reproduce dataset.py's exact sorted RGB+Max grouping before filtering NIR.
    rgb_paths = sorted(path for path, _, _ in entries.values()
                       if path.relative_to(root).as_posix() in relative_sets['rgb'])
    maximum = relative_sets['max_shadow']
    pairs = []
    for image in rgb_paths:
        mask = max_root / 'shadow_masks' / image.parent.name / image.name.replace('_rgb.png', '_shadow.png')
        if mask.relative_to(root).as_posix() in maximum:
            pairs.append((image, mask, image.parent.name))
    groups = [pair[2] for pair in pairs]
    splitter = GroupShuffleSplit(n_splits=1, test_size=options.get('val_ratio', 0.2),
                                random_state=config['random_seed'])
    training, validation = next(splitter.split(pairs, groups=groups))
    strict_keys = {row['key'] for row in records}
    report = {
        'layout': 'Extract SEO/ and segformer/ to the same parent, then cd segformer',
        'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'source_manifest_records': len(seen), 'portable_manifest_records': len(records),
        'components': components, 'paired_rgb_max': len(pairs),
        'supervision': supervision, 'random_seed': config['random_seed'],
        'val_ratio': options.get('val_ratio', 0.2), 'expected_counts': {},
        'split_key_sha256': {}, 'split_details': {},
    }
    for split, indices in (('train', training), ('val', validation)):
        strict, final, missing = [], [], []
        for index in indices:
            image, mask, _ = pairs[index]
            key = _key(image)
            if key not in strict_keys:
                continue
            strict.append(key)
            minimum = min_root / 'shadow_masks' / image.parent.name / mask.name
            uncertainty = (min_root / 'uncertainty_masks' / image.parent.name
                           / mask.name.replace('_shadow.png', '_uncertainty_mask.png'))
            complete = minimum.relative_to(root).as_posix() in relative_sets['min_shadow']
            if supervision.get('ignore_uncertainty', True):
                complete = complete and uncertainty.relative_to(root).as_posix() in relative_sets['min_uncertainty']
            (final if complete else missing).append(key)
        if missing and supervision.get('missing_policy', 'error') != 'skip':
            raise ValueError(f'Missing Min/uncertainty for {split}: {missing[:5]}')
        report['expected_counts'][split] = len(final)
        report['split_key_sha256'][split] = _keys_digest(final)
        report['split_details'][split] = {
            'rgb_max_before_nir': len(indices), 'strict_nir_before_min': len(strict),
            'after_min': len(final), 'missing_min': len(missing), 'missing_examples': missing[:5],
        }
    report['payload_files'] = len(entries)
    report['payload_bytes'] = sum(value[1] for value in entries.values())
    generated['SEO/BUNDLE_INFO.json'] = _json_bytes(report)
    return entries, generated, report


def code_plan(project):
    project = Path(project).resolve()
    entries = {}
    for name in CODE_FILES:
        path = project / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'Missing required code-bundle file: {path}')
        stat = path.stat()
        entries['segformer/' + name] = (path, stat.st_size, stat.st_mtime_ns)
    packages = {}
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python-headless',
                    'albumentations', 'scikit-learn', 'segmentation-models-pytorch',
                    'timm', 'matplotlib', 'tqdm', 'PyYAML'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    info = {'python': platform.python_version(), 'packages': packages,
            'initialization': 'Transfer the old Max-protocol best with --init-from, not --resume',
            'payload_files': len(entries), 'payload_bytes': sum(value[1] for value in entries.values())}
    return entries, {'segformer/LOCAL_ENVIRONMENT.json': _json_bytes(info)}, info


def write_archive(destination, entries, generated, progress_seconds=15):
    """Hash while writing; never stage another copy of the large image files."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to overwrite an existing archive or partial: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = sum(value[1] for value in entries.values()) + sum(map(len, generated.values()))
    minimum_space = payload_bytes + 4096 * (len(entries) + len(generated)) + 1024 ** 3
    if shutil.disk_usage(destination.parent).free < minimum_space:
        raise OSError(f'Insufficient free disk space for {destination}; need about {minimum_space} bytes')
    started, last_print, completed_bytes = time.monotonic(), 0.0, 0
    # Batch writes so a source and archive on one HDD do not alternate every few MB.
    with temporary.open('wb', buffering=256 * 1024 ** 2) as stream:
        output = DigestWriter(stream)
        with tarfile.open(fileobj=output, mode='w|', format=tarfile.PAX_FORMAT,
                          bufsize=8 * 1024 ** 2) as archive:
            archive.copybufsize = 8 * 1024 ** 2
            for number, (name, (path, size, modified)) in enumerate(sorted(entries.items()), 1):
                current = path.stat()
                if current.st_size != size or current.st_mtime_ns != modified:
                    raise RuntimeError(f'Source changed after inventory: {path}')
                header = tarfile.TarInfo(name)
                header.size, header.mode, header.mtime = size, 0o644, 0
                with path.open('rb') as source:
                    archive.addfile(header, source)
                after = path.stat()
                if after.st_size != size or after.st_mtime_ns != modified:
                    raise RuntimeError(f'Source changed during archive creation: {path}')
                completed_bytes += size
                now = time.monotonic()
                if now - last_print >= progress_seconds or number == len(entries):
                    elapsed = now - started
                    print(f'{destination.name}: {number}/{len(entries)} files; '
                          f'{completed_bytes / 1024 ** 3:.2f}/{payload_bytes / 1024 ** 3:.2f} GiB; '
                          f'{completed_bytes / max(elapsed, .001) / 1024 ** 2:.1f} MiB/s; '
                          f'{elapsed / 60:.1f} min', flush=True)
                    last_print = now
            for name, content in sorted(generated.items()):
                header = tarfile.TarInfo(name)
                header.size, header.mode, header.mtime = len(content), 0o644, 0
                archive.addfile(header, io.BytesIO(content))
        output.flush()
        checksum, size = output.digest.hexdigest(), output.bytes_written
    temporary.replace(destination)
    summary = {'file': destination.name, 'bytes': size, 'sha256': checksum,
               'members': len(entries) + len(generated)}
    destination.with_suffix(destination.suffix + '.json').write_bytes(_json_bytes(summary))
    print(json.dumps(summary), flush=True)
    return summary


def update_checksums(output):
    summaries = []
    for name in ('seo_data.tar', 'segformer_code.tar'):
        path = output / (name + '.json')
        if path.is_file() and (output / name).is_file():
            summaries.append(json.loads(path.read_text(encoding='utf-8')))
    if summaries:
        text = ''.join(f'{entry["sha256"]}  {entry["file"]}\n' for entry in summaries)
        (output / 'SHA256SUMS').write_text(text, encoding='ascii', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_seo_cca_nir.yaml')
    parser.add_argument('--output-dir', default='remote_bundle')
    parser.add_argument('--phase', choices=('data', 'code', 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    with Path(args.config).open(encoding='utf-8') as source:
        config = yaml.safe_load(source)
    for phase in (('data', 'code') if args.phase == 'all' else (args.phase,)):
        print(f'Preparing {phase} inventory...', flush=True)
        entries, generated, report = data_plan(config) if phase == 'data' else code_plan(project)
        output.mkdir(parents=True, exist_ok=True)
        (output / (phase + '_inventory.json')).write_bytes(_json_bytes(report))
        print(json.dumps(report, indent=2), flush=True)
        if not args.dry_run:
            name = 'seo_data.tar' if phase == 'data' else 'segformer_code.tar'
            write_archive(output / name, entries, generated)
            update_checksums(output)


if __name__ == '__main__':
    main()

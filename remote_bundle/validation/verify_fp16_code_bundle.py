"""Verify a replacement code tar against its inventory and current source files."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from prepare_remote_bundle import CODE_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    summary = json.loads(archive_path.with_suffix('.tar.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256()
    with archive_path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    assert archive_path.stat().st_size == summary['bytes']
    assert digest.hexdigest() == summary['sha256']
    with tarfile.open(archive_path, 'r:') as archive:
        assert set(archive.getnames()) == {
            'segformer/' + name for name in CODE_FILES
        } | {'segformer/LOCAL_ENVIRONMENT.json'}
        assert len(archive.getmembers()) == summary['members']
        for name in CODE_FILES:
            with archive.extractfile('segformer/' + name) as packed, (PROJECT / name).open('rb') as original:
                while True:
                    expected = original.read(1024 ** 2)
                    actual = packed.read(1024 ** 2)
                    assert actual == expected, name
                    if not expected:
                        break
        for name in ('config_seo_cca_nir.yaml', 'config_seo_cca_nir_server.yaml'):
            import yaml
            config = yaml.safe_load(archive.extractfile('segformer/' + name).read())
            assert config['training']['mixed_precision'] is True
            assert config['training']['batch_size'] == 6
    print(json.dumps({'verified': True, 'all_source_bytes_match': True,
                      'fp16_enabled_in_both_configs': True, **summary}, indent=2))


if __name__ == '__main__':
    main()

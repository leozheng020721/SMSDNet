"""Portable archive round-trip checks on a small synthetic SEO dataset."""

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np

from dataset import get_seo_dataset
from prepare_remote_bundle import data_plan, write_archive
import test_nir_training_flow as nir_fixture


class RemoteBundleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.source = self.temporary / 'source' / 'SEO'
        self.config = nir_fixture.NIRTrainingFlowTest.fixture(self.source, minmax=True)
        max_root = self.source / 'shadows_max/shadow_max_compressed'
        for shadow in (max_root / 'shadow_masks').rglob('*.png'):
            uncertainty = max_root / 'uncertainty_masks' / shadow.parent.name / shadow.name.replace('_shadow.png', '_uncertainty_mask.png')
            uncertainty.parent.mkdir(parents=True)
            mask = cv2.imread(str(shadow), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(uncertainty), np.zeros_like(mask))
        self.manifest = self.source / 'nir_aligned_v1/registration_with_candidates.csv'

    def test_data_tar_round_trip_preserves_split_and_relative_nir_without_source_changes(self):
        original_csv = self.manifest.read_bytes()
        manifest = list(csv.DictReader(io.StringIO(original_csv.decode())))
        # Rejected and PAN-only records must not make the strict bundle depend on their files.
        for key, status, grid in (('EXTRA/candidate', 'candidate_rgb', 'RGB'),
                                  ('EXTRA/pan', 'pan_only', 'PAN')):
            manifest.append(dict(key=key, status=status, reference_grid=grid,
                                 nir='missing_nir.tif', valid='missing_valid.png'))
        with self.manifest.open('w', newline='') as target:
            writer = csv.DictWriter(target, manifest[0])
            writer.writeheader()
            writer.writerows(manifest)
        source_csv = self.manifest.read_bytes()
        entries, generated, report = data_plan(self.config)
        self.assertEqual(report['portable_manifest_records'], 4)
        self.assertEqual(report['expected_counts'], {'train': 3, 'val': 1})
        portable = generated['SEO/nir_aligned_v1/registration_with_candidates.csv'].decode()
        for row in csv.DictReader(io.StringIO(portable)):
            for field in ('nir', 'valid'):
                self.assertNotIn(':', row[field])
                self.assertNotIn('\\', row[field])
                self.assertEqual(len(Path(row[field]).parts), 2)
        expected = {split: [Path(path).parent.name + '/' + Path(path).name
                             for path in get_seo_dataset(self.config, split).img_paths]
                    for split in ('train', 'val')}
        destination = self.temporary / 'seo_data.tar'
        summary = write_archive(destination, entries, generated)
        self.assertEqual(summary['sha256'], hashlib.sha256(destination.read_bytes()).hexdigest())
        self.assertFalse(destination.with_suffix('.tar.partial').exists())
        server = self.temporary / 'server'
        (server / 'segformer').mkdir(parents=True)
        with tarfile.open(destination) as archive:
            self.assertEqual(len(archive.getmembers()), summary['members'])
            self.assertTrue(all(member.name.startswith('SEO/') and '..' not in Path(member.name).parts
                                for member in archive))
            archive.extractall(server, filter='data')
        self.assertEqual(json.loads((server / 'SEO/BUNDLE_INFO.json').read_text())['expected_counts'], report['expected_counts'])
        remote = copy.deepcopy(self.config)
        remote['dataset']['root_dir'] = '../SEO'
        remote['dataset']['nir']['root_dir'] = '../SEO/nir_aligned_v1'
        previous = Path.cwd()
        try:
            os.chdir(server / 'segformer')
            for split in ('train', 'val'):
                dataset = get_seo_dataset(remote, split)
                actual = [Path(path).parent.name + '/' + Path(path).name for path in dataset.img_paths]
                self.assertEqual(actual, expected[split])
                sample = dataset[0]
                self.assertEqual(set(sample), {'image', 'mask', 'nir', 'nir_valid'})
                self.assertTrue((sample['mask'] == 255).any())
        finally:
            os.chdir(previous)
        self.assertEqual(self.manifest.read_bytes(), source_csv)

    def test_existing_archive_is_not_overwritten(self):
        destination = self.temporary / 'existing.tar'
        destination.write_bytes(b'existing archive')
        with self.assertRaises(FileExistsError):
            write_archive(destination, {}, {})
        self.assertEqual(destination.read_bytes(), b'existing archive')

    def test_changed_source_is_rejected(self):
        source = self.temporary / 'source.bin'
        source.write_bytes(b'original')
        stat = source.stat()
        entries = {'SEO/source.bin': (source, stat.st_size, stat.st_mtime_ns)}
        source.write_bytes(b'changed source size')
        destination = self.temporary / 'changed.tar'
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            write_archive(destination, entries, {})
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()

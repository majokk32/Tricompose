"""Authored archive/config checks only; no download, package install or parser."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chexpert_resource_setup_test', ROOT/'tools/setup_chexpert_negbio_resources.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ChexpertResourceSetupTests(unittest.TestCase):
    def test_official_upstream_constraints_preserved(self):
        source = (ROOT.parent/'chexpert-labeler/environment.yml').read_text().splitlines()
        begin = source.index('  - pip:')
        expected = [r for r in source[:begin] if r and not r.lstrip().startswith('#')]
        copied = (ROOT/'configs/chexpert_negbio_conda_exact_v1.yml').read_text().splitlines()
        self.assertEqual([r for r in copied if r and not r.lstrip().startswith('#')], expected)
        self.assertEqual((ROOT/'configs/chexpert_negbio_pip_exact_v1.txt').read_text().splitlines(),
                         [r.strip()[2:] for r in source[begin + 1:] if r.strip()])

    def test_explicit_frozen_nltk_revision(self):
        self.assertEqual(len(setup.NLTK_REV), 40)
        for name in ('punkt', 'wordnet', 'universal_tagset'):
            self.assertIn(setup.NLTK_REV, setup.ASSETS[name][0])

    def test_https_fixed_assets(self):
        for url, filename, cap in setup.ASSETS.values():
            self.assertTrue(url.startswith('https://'))
            self.assertGreater(cap, 0)
            self.assertEqual(Path(filename).name, filename)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ('../escaped', '/absolute', 'a/../../escaped', 'a\\b', '.'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    setup.destination(Path(root), path)

    def test_resource_paths_stay_inside(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(setup.destination(Path(root), 'parser/weights'), Path(root)/'parser/weights')

    def test_zip_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'authored.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('asset/data', b'authored')
            target = Path(directory)/'out'
            setup.extract_zip(archive, target)
            self.assertEqual((target/'asset/data').read_bytes(), b'authored')
            self.assertEqual((target/'asset').stat().st_mode & 0o7777, 0o2770)
            self.assertEqual((target/'asset/data').stat().st_mode & 0o7777, 0o660)

    def test_zip_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'authored.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                item = zipfile.ZipInfo('link')
                item.external_attr = 0o120777 << 16
                stream.writestr(item, 'outside')
            with self.assertRaises(ValueError):
                setup.extract_zip(archive, Path(directory)/'out')

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'authored.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('../outside', b'authored')
            with self.assertRaises(ValueError):
                setup.extract_zip(archive, Path(directory)/'out')
            self.assertFalse((Path(directory)/'outside').exists())

    def test_tar_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'authored.tar.bz2'
            with tarfile.open(archive, 'w:bz2') as stream:
                entry = tarfile.TarInfo('asset/data')
                entry.size = 8
                stream.addfile(entry, io.BytesIO(b'authored'))
            setup.extract_tar(archive, Path(directory)/'out')
            self.assertEqual((Path(directory)/'out/asset/data').read_bytes(), b'authored')

    def test_tar_links_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory)/'authored.tar.bz2'
                with tarfile.open(archive, 'w:bz2') as stream:
                    entry = tarfile.TarInfo('link')
                    entry.type = kind
                    entry.linkname = '/outside'
                    stream.addfile(entry)
                with self.assertRaises(ValueError):
                    setup.extract_tar(archive, Path(directory)/'out')

    def test_tar_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'authored.tar.bz2'
            with tarfile.open(archive, 'w:bz2') as stream:
                entry = tarfile.TarInfo('../outside')
                entry.size = 8
                stream.addfile(entry, io.BytesIO(b'authored'))
            with self.assertRaises(ValueError):
                setup.extract_tar(archive, Path(directory)/'out')

    def test_archive_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'asset'
            setup.copy_stream(io.BytesIO(b'authored'), target, 8)
            with self.assertRaises(FileExistsError):
                setup.copy_stream(io.BytesIO(b'changed!'), target, 8)
            self.assertEqual(target.read_bytes(), b'authored')

    def test_archive_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                setup.copy_stream(io.BytesIO(b'authored'), Path(directory)/'asset', 1)

    def test_non_https_download_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                setup.download('http://invalid.example/asset', Path(directory)/'asset', 10)

    def test_existing_download_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'asset'
            setup.copy_stream(io.BytesIO(b'authored'), target, 8)
            with self.assertRaises(ValueError):
                setup.download('https://invalid.example/asset', target, 10)

    def test_redirect_to_other_host_rejected(self):
        for target in ('http://www.dropbox.com/asset', 'https://untrusted.example/asset'):
            with self.assertRaises(ValueError):
                setup.TrustedRedirect().redirect_request(None, None, 302, '', {}, target)

    def test_initializer_has_no_report_input_mode(self):
        source = (ROOT/'interfaces/initialize_chexpert_negbio_legacy_v1.py').read_text()
        tree = ast.parse(source)
        arguments = [node.args[0].value for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and node.func.attr == 'add_argument' and node.args]
        self.assertEqual(arguments, ['--output'])
        self.assertNotIn('sample_reports.csv', source)
        self.assertNotIn('.parse_doc(', source)
        self.assertNotIn('.classify(', source)


if __name__ == '__main__':
    unittest.main()

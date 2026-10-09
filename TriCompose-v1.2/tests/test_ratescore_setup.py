"""Authored acquisition/approval tests; no network, weights or clinical inputs."""
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ratescore_setup_contract', ROOT / 'tools/prepare_ratescore_assets.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class Response(BytesIO):
    def geturl(self):
        return 'https://huggingface.co/public-asset'


class Opener:
    def __init__(self, payload):
        self.payload, self.requests = payload, []

    def open(self, request, *, timeout):
        self.requests.append(request)
        return Response(self.payload)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/ratescore_assets_v1.json').read_text())

    def invalid(self, mutate):
        config = deepcopy(self.config)
        mutate(config)
        with self.assertRaises(ValueError):
            setup.inventory(config)

    def test_exact_two_released_models_and_source(self):
        entries = setup.inventory(self.config)
        self.assertEqual(len(entries), 22)
        self.assertEqual(sum(e['bytes'] for e in entries if e['relative_path'].endswith('model.safetensors')),
                         1173356292)
        self.assertLess(sum(e['bytes'] for e in entries), 1300000000)
        self.assertTrue(all('/resolve/' in e['url'] or 'raw.githubusercontent.com/' in e['url'] for e in entries))

    def test_no_alternate_duplicate_weight_format(self):
        self.assertNotIn('pytorch_model.bin', {e['relative_path'] for e in setup.inventory(self.config)})

    def test_unsafe_paths(self):
        for path in ('../token', '/tmp/a', 'a/../b', 'a\\b', 'a//b', './a', 'a/./b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                setup.relative_asset(path)

    def test_duplicate_asset_rejected(self):
        self.invalid(lambda c: c['source']['files'].append(c['source']['files'][0]))

    def test_unpinned_revision_rejected(self):
        self.invalid(lambda c: c['source'].update(revision='main'))

    def test_wrong_repository_rejected(self):
        self.invalid(lambda c: c['models'][0].update(repository='not/the-official-model'))

    def test_wrong_source_rejected(self):
        self.invalid(lambda c: c['source'].update(repository='not/the-official-source'))

    def test_hash_required(self):
        self.invalid(lambda c: c['source']['files'][0].update(sha256='missing'))

    def test_byte_bounds(self):
        for value in (0, True, -1, 800000001):
            with self.subTest(value=value):
                self.invalid(lambda c: c['source']['files'][0].update(bytes=value))

    def test_cap_required(self):
        self.invalid(lambda c: c.update(maximum_asset_bytes=1))
        self.invalid(lambda c: c.update(maximum_asset_bytes=1300000001))

    def test_no_implicit_license_approval(self):
        for download, license_ok in ((False, False), (True, False), (False, True), (1, True)):
            with self.subTest(download=download, license=license_ok), self.assertRaises(ValueError):
                setup.require_approval(download, license_ok)
        setup.require_approval(True, True)

    def test_actual_job_not_environment_alone(self):
        setup.require_slurm('3:memory:/slurm/uid_1/job_123/step_batch\n', '123')
        for cgroup, job in (('3:memory:/\n', '123'), ('/job_1234/task_0', '123'),
                            ('/job_123/task_0', '12/3'), ('/job_123/task_0', '')):
            with self.subTest(cgroup=cgroup, job=job), self.assertRaises(ValueError):
                setup.require_slurm(cgroup, job)

    def test_existing_runtime_parent_mode_preserved(self):
        with tempfile.TemporaryDirectory(dir=setup.WORKSPACE / '.tmp') as directory:
            workspace = Path(directory)
            (workspace / 'runtime').mkdir(mode=0o2770)
            path = workspace / 'runtime/models'
            path.mkdir(mode=0o2750)
            path.chmod(0o2750)
            before = (path.stat().st_mode, path.stat().st_gid)
            with patch.object(setup, 'WORKSPACE', workspace):
                setup.private_dir(path, allow_existing_readonly=True)
            self.assertEqual((path.stat().st_mode, path.stat().st_gid), before)

    def fetch(self, payload, expected=b'public model asset', before=None):
        with tempfile.TemporaryDirectory(dir=setup.WORKSPACE / '.tmp') as directory:
            target = Path(directory) / 'model.safetensors'
            if before:
                before(target)
            opener = Opener(payload)
            entry = {'url': 'https://huggingface.co/repo/resolve/fixed/model.safetensors',
                     'bytes': len(expected), 'sha256': hashlib.sha256(expected).hexdigest()}
            setup.fetch(entry, target, opener)
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(target.stat().st_mode & 0o777, 0o660)
            self.assertFalse(target.with_name(target.name + '.partial').exists())
            self.assertFalse(any(k.lower() == 'authorization' for k in opener.requests[0].headers))

    def test_verified_stream_without_credentials(self):
        self.fetch(b'public model asset')

    def test_oversized_stream_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch(b'public model asset extra')

    def test_short_stream_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch(b'public')

    def test_wrong_hash_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch(b'PUBLIC MODEL ASSET')

    def test_no_existing_file_overwrite(self):
        with self.assertRaises(ValueError):
            self.fetch(b'public model asset', before=lambda p: p.write_bytes(b'old'))

    def test_no_symlink_target(self):
        with self.assertRaises(ValueError):
            self.fetch(b'public model asset', before=lambda p: p.symlink_to('absent-target'))

    def test_batch_has_approval_and_no_inference(self):
        script = (ROOT / 'slurm/61_ratescore_setup_cpu.sbatch').read_text()
        self.assertIn('TRICOMPOSE_APPROVE_DOWNLOADS', script)
        self.assertIn('TRICOMPOSE_CONFIRM_BIOLORD_LICENSE', script)
        self.assertIn('--partition=main', script)
        self.assertNotIn('--gres', script)
        self.assertNotIn('sbatch ', script)
        self.assertNotIn('compute_score(', script)
        self.assertNotIn('torchrun', script)


if __name__ == '__main__':
    unittest.main()

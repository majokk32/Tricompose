"""Fake CPU interface and invented texts only. No torch/model/patient reads."""
import contextlib
import copy
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_manual_opacity_dev_chexbert as worker


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeModel:
    def __init__(self, *, maximum=512, output=None, fail=False):
        self.bert = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=maximum))
        self.output = output
        self.fail = fail
        self.calls = []
        self.preflight = []

    def tokenizer(self, text, *, truncation):
        self.preflight.append((text, truncation))
        return {'input_ids': list(range(len(text.split()) + 2))}

    def __call__(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError('intentionally authored private fake failure')
        return FakeTensor(self.output if self.output is not None else [[0, 0, 1] + [0] * 11 for _ in texts])


def requests(count):
    docs = [{'sentences': [[f'Fictional-{i}', 'opacity', 'example.']],
             'ner': ['deliberately-invalid-not-read'], 'doc_key': 'fictional-key-must-not-export'} for i in range(count)]
    return worker.requests_from_documents(docs, count)


def predict(model, texts, rows, maximum=75):
    return worker.predict_requests(model, texts, rows, contextlib.nullcontext,
        batch_size=4, maximum_examples=maximum, public_stdout=io.StringIO())


class ManualOpacityReaderWorkerTests(unittest.TestCase):
    def test_requests_do_not_use_annotations_or_source_keys(self):
        doc = {'sentences': [['Fully', 'invented', 'opacity.']], 'ner': [['unsupported']], 'doc_key': 'never-output-key'}
        texts, rows = worker.requests_from_documents([doc], 1)
        other = copy.deepcopy(doc)
        other['ner'] = object()
        self.assertEqual((texts, rows), worker.requests_from_documents([other], 1))
        self.assertNotIn('never-output-key', json.dumps(rows))
        self.assertNotIn('invented', json.dumps(rows))
        self.assertEqual(rows[0]['source_sha256'], hashlib.sha256(texts['report_0000'].encode()).hexdigest())

    def test_complete_official_fourteen_heads_all_slots_attempted(self):
        texts, rows = requests(5)
        fake = FakeModel()
        snapshot = copy.deepcopy(texts)
        counters = predict(fake, texts, rows)
        self.assertEqual(counters['encoder_examples'], 5)
        self.assertEqual(counters['forward_batches'], 2)
        self.assertEqual([len(c) for c in fake.calls], [4, 1])
        self.assertTrue(all(r['status'] == 'complete' and len(r['finding_states']) == 14 for r in rows))
        self.assertTrue(all(r['finding_states']['lung_opacity'] == 'positive' for r in rows))
        self.assertEqual(texts, snapshot)

    def test_overflow_retained_without_truncation_or_forward(self):
        texts, rows = requests(1)
        fake = FakeModel(maximum=4)
        counters = predict(fake, texts, rows)
        self.assertEqual(counters['encoder_examples'], 0)
        self.assertEqual(fake.calls, [])
        self.assertFalse(fake.preflight[0][1])
        self.assertEqual(rows[0]['failure_type'], 'token_budget_exceeded_no_truncation')
        self.assertIsNone(rows[0]['finding_states'])

    def test_exact_token_limit_allowed(self):
        texts, rows = requests(1)
        fake = FakeModel(maximum=5)
        self.assertEqual(predict(fake, texts, rows)['encoder_examples'], 1)
        self.assertEqual(rows[0]['status'], 'complete')

    def test_failed_batch_never_retried_or_dropped(self):
        texts, rows = requests(5)
        fake = FakeModel(fail=True)
        self.assertEqual(predict(fake, texts, rows)['encoder_examples'], 5)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(r['status'] == 'failed_unavailable' and r['failure_type'] == 'RuntimeError' for r in rows))
        self.assertNotIn('authored private', json.dumps(rows))

    def test_invalid_batch_has_no_partial_success(self):
        texts, rows = requests(2)
        fake = FakeModel(output=[[0, 0, 1] + [0] * 11, [0] * 13])
        predict(fake, texts, rows)
        self.assertTrue(all(r['status'] == 'failed_unavailable' for r in rows))
        self.assertTrue(all(r['finding_states'] is None for r in rows))

    def test_unreadable_text_kept_as_failed_slot(self):
        texts, rows = worker.requests_from_documents([{'sentences': [[None]]}], 1)
        self.assertEqual(texts, {})
        self.assertEqual(rows[0]['status'], 'failed_unavailable')
        self.assertEqual(predict(FakeModel(), texts, rows)['encoder_examples'], 0)
        self.assertEqual(rows[0]['failure_type'], 'ValueError')

    def test_budget_change_or_shortened_source_forbidden(self):
        texts, rows = requests(2)
        with self.assertRaises(ValueError):
            predict(FakeModel(), texts, rows, maximum=1)
        with self.assertRaises(ValueError):
            worker.requests_from_documents([{'sentences': [['Invented']]}], 2)

    def test_request_dependencies_exist_and_official_asset_paths_not_changed(self):
        config = json.loads(worker.CONFIG.read_text())
        worker.verify_request(config)
        self.assertTrue(all(p.is_file() for p in worker.sources()))
        self.assertEqual(config['attempted_reports'], 75)
        self.assertFalse(config['download_allowed'])
        self.assertFalse(config['regeneration_authorized'])
        self.assertEqual(worker.CHECKPOINT.name, 'chexbert.pth')
        # Unchanged adapter joins cache_dir with the absolute checkpoint path.
        self.assertEqual(worker.os.path.join('/fictional-protected-cache', str(worker.CHECKPOINT)), str(worker.CHECKPOINT))
        for field, value in (('retries', 1), ('device', 'cuda'), ('batch_size', 8), ('attempted_reports', 74)):
            altered = dict(config, **{field: value})
            with self.assertRaises(ValueError):
                worker.verify_request(altered)

    def test_current_cache_only_allocation_cannot_load_source_or_model(self):
        env = {'SLURM_JOB_ID': '12784259', 'TRICOMPOSE_MANUAL_OPACITY_READER': 'manual-opacity-chexbert-request-v1'}
        with patch.object(worker.sys, 'argv', ['worker']), patch.object(worker.os, 'environ', env), \
             patch.object(worker.Path, 'read_text', return_value='/slurm/job_12784259/step_batch'), \
             patch.object(worker, 'execute') as execute, patch.object(worker, 'new_atomic_run') as new_run, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()
            new_run.assert_not_called()

    def test_direct_worker_call_also_refuses_cache_only_allocation(self):
        env = {'SLURM_JOB_ID': '12784259', 'TRICOMPOSE_MANUAL_OPACITY_READER': 'manual-opacity-chexbert-request-v1'}
        with patch.object(worker.sys, 'argv', ['worker']), patch.object(worker.os, 'environ', env), \
             patch.object(worker.Path, 'read_text', return_value='/slurm/job_12784259/step_batch'), \
             patch.object(worker, 'sha256') as sha:
            with self.assertRaises(ValueError):
                worker.execute(None, io.StringIO())
            sha.assert_not_called()

    def test_login_gpu_scope_or_argument_failure_precedes_any_source_or_model(self):
        marker = {'TRICOMPOSE_MANUAL_OPACITY_READER': 'manual-opacity-chexbert-request-v1'}
        attempts = [(['worker'], marker, '/login'),
            (['worker'], {'SLURM_JOB_ID': '123'}, '/slurm/job_123/step_batch'),
            (['worker', '--choose-easy'], dict(marker, SLURM_JOB_ID='123'), '/slurm/job_123/step_batch'),
            (['worker'], dict(marker, SLURM_JOB_ID='123', SLURM_JOB_GPUS='0'), '/slurm/job_123/step_batch')]
        for argv, env, cgroup in attempts:
            with patch.object(worker.sys, 'argv', argv), patch.object(worker.os, 'environ', env), \
                 patch.object(worker.Path, 'read_text', return_value=cgroup), \
                 patch.object(worker, 'execute') as execute, patch.object(worker, 'new_atomic_run') as new_run, patch('builtins.print'):
                self.assertEqual(worker.main(), 1)
                execute.assert_not_called()
                new_run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

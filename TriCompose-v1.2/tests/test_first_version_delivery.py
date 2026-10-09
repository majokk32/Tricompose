"""Invented metadata fixtures only; no patient/model or generated payloads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('first_version_delivery_fixture',
    ROOT / 'tools/build_first_version_delivery.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    scores, cache, images, reports, inputs = [], {}, {}, {}, {}
    case = 'fixture_case_0'
    for model in sorted(w.CXR_MODELS):
        iid = 'image_' + model
        images[iid] = {'case_id': case, 'candidate_id': iid, 'model_id': model, 'seed': 0,
            'artifact': {'path': '/invented/' + iid + '.png', 'sha256': h(iid)},
            'prompt_sha256': h('prompt_' + model), 'frozen_model': True}
        inputs[case, model, 0] = {
            'synthetic_ehr': {'path': '/invented/ehr.json', 'sha256': h('ehr')},
            'ehr_facts': {'path': '/invented/facts.json', 'sha256': h('facts')},
            'final_prompt': {'path': '/invented/' + model + '.txt', 'sha256': h('prompt_' + model)}}
        for expert in sorted(w.REPORT_MODELS):
            rid, cid = 'report_' + model + '_' + expert, 'triple_' + model + '_' + expert
            reports[rid] = {'case_id': case, 'candidate_id': rid, 'model_id': expert,
                'parent_cxr_candidate_id': iid, 'artifact': {'path': '/invented/' + rid + '.txt', 'sha256': h(rid)},
                'frozen_model': True, 'structured_ehr_content_supplied_to_model': False,
                'source_report_or_real_target_supplied': False}
            lineage = {'cxr_model_id': model, 'cxr_seed': 0, 'report_model_id': expert,
                'cxr_candidate_id': iid, 'report_candidate_id': rid, 'ehr_sha256': h('ehr'),
                'ehr_facts_sha256': h('facts'), 'cxr_sha256': h(iid), 'report_sha256': h(rid)}
            score = {'case_id': case, 'triple_candidate_id': cid, 'lineage': lineage,
                'scoring': {'selection': {'selected': not scores},
                    'modality_quality': {'report_structure_quality_score_0_1': .6},
                    'cost': {'known_runtime_seconds': None}}}
            row = {k: v for k, v in lineage.items() if k not in ('cxr_model_id', 'cxr_seed')}
            row.update(case_id=case, triple_candidate_id=cid,
                source_primary_prefix='[0,0,0,-0.5]', biovil_raw_cosine='0.5')
            for edge in w.cached.EDGES:
                values = dict(known_reference_facts=1, comparable_facts=1, supported_facts=1,
                    supported_positive=1, supported_negative=0, proxy_opposition_facts=0)
                row.update({edge + '_' + k: str(v) for k, v in values.items()})
            scores.append(score); cache[cid] = row
    return scores, cache, images, reports, inputs


def project(values=None):
    return w.project_index(*(values or fixture()), expected_cases=1)


class FirstVersionDeliveryTests(unittest.TestCase):
    def test_exact_three_by_four_grid_and_paths(self):
        index = project()
        self.assertEqual(len(index), 12)
        info = w.inventory(index)
        self.assertEqual(info['fixed_ehr_cases'], 1)
        self.assertEqual(info['cxr_candidate_slots'], 3)
        self.assertEqual(info['report_candidate_slots'], 12)
        self.assertEqual(info['historical_selected_triples'], 1)
        self.assertFalse(info['clinical_best_triple_established'])
        self.assertFalse(info['native_structured_ehr_to_cxr'])
        self.assertFalse(info['genuine_ehr_plus_cxr_report_path'])

    def test_order_deterministic_and_source_not_mutated(self):
        values = fixture(); before = copy.deepcopy(values)
        a = project(values)
        values[0].reverse()
        b = project(values)
        self.assertEqual(a, b)
        values[0].reverse()
        self.assertEqual(values, before)

    def test_extra_source_fields_not_exported(self):
        values = fixture()
        for score in values[0]:
            score['report_text'] = object()
            score['lineage']['raw_ehr'] = object()
        for image in values[2].values():
            image['pixel_contents'] = object()
        for report in values[3].values():
            report['report_text'] = object()
        text = json.dumps(project(values))
        for key in ('report_text', 'raw_ehr', 'pixel_contents'):
            self.assertNotIn(key, text)

    def test_missing_duplicate_and_changed_seed_refused(self):
        values = fixture(); values[0].pop()
        with self.assertRaises(ValueError): project(values)
        values = fixture(); values[0][1] = copy.deepcopy(values[0][0])
        with self.assertRaises(ValueError): project(values)
        values = fixture(); values[0][0]['lineage']['cxr_seed'] = 1
        with self.assertRaises(ValueError): project(values)

    def test_changed_ehr_or_endpoint_hash_refused(self):
        for field in ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256'):
            values = fixture(); values[0][0]['lineage'][field] = h('changed')
            with self.assertRaises(ValueError): project(values)

    def test_parent_case_model_prompt_and_frozen_scope_refused(self):
        for mode in ('parent', 'case', 'model', 'prompt', 'frozen', 'ehr_supplied', 'target_supplied'):
            values = fixture(); score = values[0][0]
            report = values[3][score['lineage']['report_candidate_id']]
            image = values[2][score['lineage']['cxr_candidate_id']]
            if mode == 'parent': report['parent_cxr_candidate_id'] = 'other_image'
            if mode == 'case': report['case_id'] = 'other_case'
            if mode == 'model': report['model_id'] = 'other_model'
            if mode == 'prompt': image['prompt_sha256'] = h('changed')
            if mode == 'frozen': report['frozen_model'] = False
            if mode == 'ehr_supplied': report['structured_ehr_content_supplied_to_model'] = True
            if mode == 'target_supplied': report['source_report_or_real_target_supplied'] = True
            with self.assertRaises(ValueError): project(values)

    def test_unknown_ehr_edges_remain_null_and_no_disease_inserted(self):
        values = fixture()
        for row in values[1].values():
            for edge in ('ehr_cxr', 'ehr_report'):
                for field in w.cached.COUNT_FIELDS:
                    row[edge + '_' + field] = '0'
        index = project(values)
        self.assertTrue(all(r['ehr_cxr_support_over_known'] is None for r in index))
        self.assertTrue(all(r['ehr_report_coverage_over_known'] is None for r in index))
        self.assertEqual(w.inventory(index)['cases_without_direct_cached_ehr_facts'], 1)

    def test_selected_flag_is_preserved_not_recomputed_from_cosine(self):
        values = fixture(); selected = values[0][0]['triple_candidate_id']
        values[1][selected]['biovil_raw_cosine'] = '-0.9'
        index = project(values)
        self.assertEqual([r['triple_candidate_id'] for r in index if r['historical_static_selected']], [selected])
        values[0][1]['scoring']['selection']['selected'] = True
        with self.assertRaises(ValueError): project(values)

    def test_exact_csv_comparison_replay(self):
        expected = [{'method': 'fixed', 'metric': None}, {'method': 'static_rerank', 'metric': .5}]
        actual = list(w.csv.DictReader(w.io.StringIO(w.cached.csv_text(expected))))
        w.check_records_equal(actual, expected, 'invented_mismatch')
        actual[0]['metric'] = '0'
        with self.assertRaises(ValueError): w.check_records_equal(actual, expected, 'invented_mismatch')

    def test_case_index_retains_missing_baseline_instead_of_drop(self):
        index = project(); cid = index[0]['triple_candidate_id']
        old = [{'case_id': 'fixture_case_0', 'method': method, 'model_call_budget': 30,
                'selected_candidate_id': None if method == 'fixed' else cid}
            for method in ('fixed', 'static_rerank')]
        result = w.case_index(index, old)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]['fixed_candidate_id'])
        self.assertIsNone(result[0]['fixed_report_path'])
        self.assertEqual(result[0]['static_rerank_report_path'], index[0]['report_path'])

    def test_cross_case_baseline_choice_refused(self):
        old = [{'case_id': 'fixture_case_0', 'method': method, 'model_call_budget': 30,
            'selected_candidate_id': 'foreign_id'} for method in ('fixed', 'static_rerank')]
        with self.assertRaises(ValueError): w.case_index(project(), old)

    def test_duplicate_report_hashes_visible_not_discarded(self):
        index = project()
        for row in index: row['report_sha256'] = h('same_invented_report')
        info = w.inventory(index)
        self.assertEqual(info['report_candidate_slots'], 12)
        self.assertEqual(info['distinct_report_hashes'], 1)

    def test_available_cases_not_inflated_by_candidate_slots(self):
        rows = w.availability(project())
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row['fixed_ehr_cases'], 1)
            self.assertEqual(row['candidate_slots'], 12)
            self.assertEqual(row['cases_with_known_reference_facts'], 1)
            self.assertEqual(row['reference_observations_repeated_across_candidates'], 12)

    def test_guard_runs_before_loading(self):
        with patch.object(w.cached, 'cpu_guard', side_effect=RuntimeError('invented_guard')), \
                patch.object(w.cached, 'load') as loader:
            with self.assertRaises(RuntimeError): w.execute(Path('/invented'), 'new')
            loader.assert_not_called()

    def test_entry_report_and_dictionary_disclose_cost_and_limits(self):
        text = w.render(w.inventory(project()), [])
        for fragment in ('not native structured EHR', 'CXRMate-ED', '0/2', 'NA',
                         'NOT image/report/EHR payloads', 'All five caps'):
            self.assertIn(fragment, text)
        self.assertIn('not measured GPU', w.DICTIONARY)
        self.assertIn('Missing costs stay NA', w.DICTIONARY)


if __name__ == '__main__':
    unittest.main()

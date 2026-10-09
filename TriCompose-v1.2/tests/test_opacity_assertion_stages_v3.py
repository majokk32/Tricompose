"""Format-only checks; no inference or reinterpretation of failed V2 outputs."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'interfaces'), str(ROOT / 'benchmarks')]
sys.path.insert(0, str(ROOT / 'tools'))
import opacity_assertion_stages_v2 as old
import opacity_assertion_stages_v3 as new
import opacity_assertion_controls_v2 as fixtures
import benchmark_opacity_assertion_stages_v2 as old_worker
import benchmark_opacity_assertion_stages_v3 as new_worker


class FormatOnlyTests(unittest.TestCase):
    def test_version_distinct(self):
        self.assertNotEqual(old.VERSION, new.VERSION)

    def test_empty_instruction_matches_object_decoder(self):
        self.assertNotIn('return [].', new.LOCATOR_PROMPT)
        self.assertIn('return {"segment_ids": []}.', new.LOCATOR_PROMPT)
        self.assertIn('never a bare array', new.LOCATOR_PROMPT)

    def test_only_two_declared_prompt_replacements(self):
        expected = old.LOCATOR_PROMPT.replace('If no segment discusses opacity, return [].',
            'If no segment discusses opacity, return {"segment_ids": []}.').replace(
            'Use only the supplied segment IDs. No quotes, states, explanations or extra keys.',
            'Your entire response must be the object with segment_ids, never a bare array.\n'
            'Use only the supplied segment IDs. No quotes, states, explanations or extra keys.')
        self.assertEqual(new.LOCATOR_PROMPT, expected)

    def test_old_locator_prompt_stays_unchanged(self):
        self.assertIn('return [].', old.LOCATOR_PROMPT)

    def test_same_polarity_prompt_and_functions(self):
        self.assertEqual(new.POLARITY_PROMPT, old.POLARITY_PROMPT)
        for name in ('segments', 'validate_inventory', 'polarity_messages', 'decode_locator',
                     'decode_polarity', 'reduce_states', 'evidence'):
            self.assertIs(getattr(new, name), getattr(old, name))

    def test_bare_array_still_rejected_not_rescued(self):
        for value in ('[]', '[0]'):
            with self.assertRaises(ValueError): new.decode_locator(value, new.segments('Lung opacity.'))

    def test_valid_empty_and_nonempty_objects(self):
        inventory = new.segments('Lung opacity.')
        self.assertEqual(new.decode_locator('{"segment_ids":[]}', inventory), [])
        self.assertEqual(new.decode_locator('{"segment_ids":[0]}', inventory), [0])

    def test_actual_message_uses_corrected_final_prompt(self):
        text = 'Invented {segments} text.'
        messages = new.locator_messages(text, new.segments(text))
        rendered = messages[0]['content'][0]['text']
        self.assertIn(text, rendered)
        self.assertIn('never a bare array', rendered)
        self.assertNotIn('return [].', rendered)

    def test_all_48_inputs_and_source_spans_unchanged(self):
        self.assertEqual(len(fixtures.cases()), 48)
        for row in fixtures.cases():
            self.assertEqual(new.segments(row['text']), old.segments(row['text']))

    def test_no_reference_or_case_metadata_in_messages(self):
        for row in fixtures.cases():
            messages = new.locator_messages(row['text'], new.segments(row['text']))
            self.assertEqual([c['type'] for c in messages[0]['content']], ['text'])
            for key in ('expected_state', 'legacy_control_id', 'item_id', 'family'):
                self.assertNotIn(key, messages[0]['content'][0]['text'])

    def test_worker_only_declared_version_binding_changes(self):
        source = (ROOT / 'tools/benchmark_opacity_assertion_stages_v2.py').read_text()
        expected = source.replace('import opacity_assertion_stages_v2 as stages', 'import opacity_assertion_stages_v3 as stages').replace(
            "SCHEMA = 'tricompose-authored-opacity-two-stage-v2'", "SCHEMA = 'tricompose-authored-opacity-two-stage-v3-format-only'").replace(
            'tests/test_opacity_assertion_stages_v2.py', 'tests/test_opacity_assertion_stages_v3.py').replace(
            'docs/opacity_assertion_stages_v2_protocol.md', 'docs/opacity_assertion_stages_v3_protocol.md').replace(
            "'retry_failed': False, 'input_source':", "'format_only_fix_after_v2_result': True, 'retry_failed': False, 'input_source':").replace(
            'paths = [Path(__file__), Path(stages.__file__), FIXTURES, TESTS, PROTOCOL,',
            "paths = [Path(__file__), Path(stages.__file__),\n        ROOT / 'interfaces/opacity_assertion_stages_v2.py',\n        ROOT / 'tools/benchmark_opacity_assertion_stages_v2.py', FIXTURES, TESTS, PROTOCOL,")
        self.assertEqual((ROOT / 'tools/benchmark_opacity_assertion_stages_v3.py').read_text().rstrip(), expected.rstrip())

    def test_same_clinical_cost_fixture_policy(self):
        policy = {k: v for k, v in new_worker.POLICY.items() if k != 'format_only_fix_after_v2_result'}
        self.assertEqual(policy, old_worker.POLICY)
        self.assertIs(new_worker.stages, new)
        self.assertIs(old_worker.stages, old)
        self.assertEqual(new_worker.FIXTURES, old_worker.FIXTURES)
        self.assertNotEqual(new_worker.SCHEMA, old_worker.SCHEMA)


if __name__ == '__main__':
    unittest.main()

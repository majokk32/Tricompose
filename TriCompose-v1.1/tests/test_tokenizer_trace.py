import unittest
from types import SimpleNamespace

from tricompose_v11.tokenizer_trace import observe_pipeline_tokenizer, text_sha256
from tricompose_v11.tokenizer_audit import reconstruct_text, sana_instruction


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.last_result = None
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, dict(kwargs)))
        texts = [text] if isinstance(text, str) else text
        ids = [[ord(c) for c in s] for s in texts]
        self.last_result = {"input_ids": ids, "attention_mask": [[1] * len(s) for s in texts]}
        return self.last_result

    def encode(self, text):
        return [ord(c) for c in text]


class RegisteredPipeline:
    def __init__(self, tokenizer):
        object.__setattr__(self, "registration_changes", 0)
        object.__setattr__(self, "tokenizer", tokenizer)

    def __setattr__(self, name, value):
        object.__setattr__(self, "registration_changes", self.registration_changes + 1)
        object.__setattr__(self, name, value)


class TokenizerTraceTests(unittest.TestCase):
    def test_proxy_preserves_result_identity_arguments_and_registration(self):
        base = FakeTokenizer()
        pipe = RegisteredPipeline(base)
        runtime = SimpleNamespace(pipe=pipe)
        prompts = [" FINDINGS. ", "Other Condition"]
        with observe_pipeline_tokenizer(runtime) as trace:
            pipe.tokenizer.padding_side = "right"
            self.assertEqual(base.padding_side, "right")
            self.assertEqual(pipe.tokenizer.encode("prefix"), base.encode("prefix"))
            pipe.tokenizer(prompts[0], truncation=False)  # Length check, ignored.
            final = ["OFFICIAL PREFIX: " + p.lower().strip() for p in prompts]
            result = pipe.tokenizer(final, padding="max_length", max_length=300, truncation=True, return_tensors="pt")
            self.assertIs(result, base.last_result)
            pipe.tokenizer(["", ""], padding="max_length", return_tensors="pt")
        self.assertIs(pipe.tokenizer, base)
        self.assertEqual(pipe.registration_changes, 0)
        payload = trace.candidate_payload(1, prompts)
        self.assertTrue(payload["runtime_observed"])
        self.assertTrue(payload["pipeline_changed_text"])
        self.assertEqual(payload["positive_tokenizer_text"], final[1])
        self.assertEqual(payload["positive_tokenizer_text_sha256"], text_sha256(final[1]))
        self.assertEqual(payload["attention_token_count"], len(final[1]))
        self.assertEqual(len(payload["tensor_tokenizer_calls"]), 2)

    def test_restored_on_pipeline_failure(self):
        base = FakeTokenizer()
        runtime = SimpleNamespace(pipe=RegisteredPipeline(base))
        with self.assertRaises(RuntimeError):
            with observe_pipeline_tokenizer(runtime):
                raise RuntimeError("synthetic failure")
        self.assertIs(runtime.pipe.tokenizer, base)

    def test_missing_or_misaligned_capture_fails(self):
        runtime = SimpleNamespace(pipe=RegisteredPipeline(FakeTokenizer()))
        with observe_pipeline_tokenizer(runtime) as trace:
            with self.assertRaises(ValueError):
                trace.positive_batch(["expected"])
            runtime.pipe.tokenizer(["wrong"], padding="max_length", return_tensors="pt")
        with self.assertRaises(ValueError):
            trace.positive_batch(["expected"])

    def test_no_pipeline_is_not_silently_marked_observed(self):
        with self.assertRaises(ValueError):
            with observe_pipeline_tokenizer(SimpleNamespace()):
                pass

    def test_reconstruction_keeps_official_prefix_exactly_once(self):
        source = 'class SanaPipeline:\n    def __call__(self, complex_human_instruction=["Official", "User: "]):\n        pass\n'
        prefix = sana_instruction(source)
        self.assertEqual(prefix, "Official\nUser: ")
        self.assertEqual(reconstruct_text("chexgenbench_sana", " X Ray ", instruction=prefix), "Official\nUser: x ray")
        self.assertEqual(reconstruct_text("chexgenbench_pixart", " X Ray "), "x ray")
        self.assertEqual(reconstruct_text("roentgen_v2", " X Ray "), " X Ray ")

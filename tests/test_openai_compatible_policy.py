from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tricompose.agent.openai_compatible_policy import (
    _call_api,
    _parse_llm_decision,
    guard_llm_decision,
    sanitize_agent_state,
)


class OpenAICompatiblePolicyTests(unittest.TestCase):
    def test_openai_compatible_http_round_trip(self) -> None:
        class StubHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers["Content-Length"])
                request_payload = json.loads(
                    self.rfile.read(content_length).decode("utf-8")
                )
                self.server.seen_model = request_payload["model"]
                response_payload = {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"action":"verify_more",'
                                    '"reason_code":"demo_check"}'
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 42},
                }
                encoded = json.dumps(response_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            decision, usage = _call_api(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="synthetic-model",
                api_key=None,
                state={
                    "state_id": "synthetic_demo",
                    "candidate_scores": {"a": 0.7, "b": 0.5},
                    "peer_metrics": {},
                    "policy": {
                        "minimum_score": 0.55,
                        "minimum_margin": 0.10,
                    },
                    "allowed_actions": ["select", "verify_more"],
                },
                timeout_seconds=2,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(decision["action"], "verify_more")
        self.assertEqual(usage, {"total_tokens": 42})
        self.assertEqual(server.seen_model, "synthetic-model")

    def test_sanitizer_drops_paths_hashes_and_candidate_metadata(self) -> None:
        source = {
            "schema_version": "tricompose.agent_decision.v1",
            "run_id": "protected_run",
            "candidates": {
                "a": {
                    "artifact_sha256": "f" * 64,
                    "path": "/must/not/leave",
                }
            },
            "evidence": {
                "candidate_scores": {"a": 0.8, "b": 0.4},
                "peer_metrics": {"report_pair_bleu_1": 0.1},
                "candidate_score_bundle_sha256": "e" * 64,
            },
            "policy": {
                "minimum_score": 0.55,
                "minimum_margin": 0.10,
            },
        }

        sanitized = sanitize_agent_state(source)

        rendered = str(sanitized)
        self.assertNotIn("artifact_sha256", rendered)
        self.assertNotIn("/must/not/leave", rendered)
        self.assertNotIn("f" * 64, rendered)
        self.assertEqual(sanitized["candidate_scores"]["a"], 0.8)

    def test_parser_accepts_json_only_decision(self) -> None:
        decision = _parse_llm_decision(
            '{"action":"select","candidate_id":"a",'
            '"reason_code":"best_score"}'
        )

        self.assertEqual(decision["action"], "select")
        self.assertEqual(decision["candidate_id"], "a")

    def test_guard_blocks_selection_when_threshold_fails(self) -> None:
        state = {
            "candidate_scores": {"a": 0.4, "b": 0.3},
            "policy": {
                "minimum_score": 0.55,
                "minimum_margin": 0.10,
            },
        }

        guarded = guard_llm_decision(
            {
                "action": "select",
                "candidate_id": "a",
                "reason_code": "best_score",
            },
            state,
        )

        self.assertEqual(guarded["action"], "verify_more")
        self.assertEqual(guarded["status"], "guard_override")


if __name__ == "__main__":
    unittest.main()

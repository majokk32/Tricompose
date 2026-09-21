from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tricompose_v11.cxr_execution as execution


class V11CXRExecutionCleanupTests(unittest.TestCase):
    def test_preexisting_target_is_never_deleted_on_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tricompose_v11_cxr_") as temp:
            root = Path(temp)
            request = root / "requests"
            output = root / "outputs"
            request.mkdir()
            output.mkdir()
            target = output / "same_run_id"
            target.mkdir()
            sentinel = target / "sentinel"
            sentinel.write_text("committed")
            with (
                mock.patch.object(execution, "MAIN_PROTECTED_ROOT", root),
                mock.patch.object(
                    execution,
                    "load_model_requests",
                    return_value=[],
                ),
            ):
                with self.assertRaises(FileExistsError):
                    execution.run_cxr_request_run(
                        request_run=request,
                        output_root=output,
                        output_run_id="same_run_id",
                        model_id="chexgenbench_pixart",
                        model_revision="revision",
                        model_audit={},
                        runtime_factory=lambda: None,
                        batch_size=1,
                    )
            self.assertEqual(sentinel.read_text(), "committed")


if __name__ == "__main__":
    unittest.main()

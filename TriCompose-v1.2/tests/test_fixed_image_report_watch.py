"""Mocked clocks/states only; never wait in real time or spawn processes."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixed_report_observer", ROOT/"audits/watch_fixed_image_reports.py")
observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)


class WatchTests(unittest.TestCase):
    def call(self, states, manifest=True, maximum=121):
        clock = [0.0]; delays = []
        def sleep(value): delays.append(value); clock[0] += value
        state = Mock(side_effect=states)
        result = observer.wait_for_completion(state, lambda: manifest, max_wait=maximum, poll_seconds=60,
            now=lambda: clock[0], sleep=sleep)
        return result, delays

    def test_pending_then_complete_audits_only_after_completion(self):
        value, delays = self.call(["PENDING", "RUNNING", "COMPLETED"])
        self.assertEqual(value, "ready_for_metadata_audit"); self.assertEqual(delays, [60,60])

    def test_manifest_alone_cannot_trigger_audit_while_running(self):
        value, _ = self.call(["RUNNING"]*5)
        self.assertEqual(value, "observer_deadline_expired_no_audit")

    def test_missing_manifest_not_completion_success(self):
        value, delays = self.call(["COMPLETED"], manifest=False)
        self.assertEqual(value, "completed_without_published_manifest"); self.assertFalse(delays)

    def test_failed_jobs_do_not_get_retried_or_audited(self):
        for state in observer.TERMINAL_FAILURES:
            value, delays = self.call([state])
            self.assertEqual(value, "gpu_job_terminal_without_audit"); self.assertFalse(delays)

    def test_unknown_accounting_is_bounded_and_not_busy_polled(self):
        value, delays = self.call(["UNKNOWN"]*5)
        self.assertEqual(value, "observer_deadline_expired_no_audit"); self.assertEqual(delays, [60,60,1.0])

    def test_invalid_bounds_refused_before_status_access(self):
        for maximum, period in ((True,60), (0,60), (10801,60), (1,0), (1,61)):
            status=Mock()
            with self.assertRaises(ValueError): observer.wait_for_completion(status, Mock(), max_wait=maximum, poll_seconds=period)
            status.assert_not_called()


if __name__ == "__main__": unittest.main()

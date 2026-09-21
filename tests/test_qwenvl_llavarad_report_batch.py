from tricompose.verifiers.qwenvl_llavarad_report_batch import LLAVARAD_SPEC


def test_llavarad_report_contract() -> None:
    assert LLAVARAD_SPEC.candidate_id == "llavarad"
    assert LLAVARAD_SPEC.namespace == "llavarad_report"
    assert LLAVARAD_SPEC.run_schema == "tricompose.llavarad_report.run.v1"
    assert LLAVARAD_SPEC.frozen_schema == "tricompose.llavarad_report.frozen_run.v1"

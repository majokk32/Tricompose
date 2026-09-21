from tricompose.verifiers.qwenvl_cxr_report_batch import deterministic_flags


def test_deterministic_flags_temporal_and_measurement() -> None:
    flags = deterministic_flags(
        "Compared to the prior study, a 2.4 cm opacity is unchanged."
    )
    assert flags["unsupported_temporal_language"] is True
    assert flags["exact_cm_mm_measurement"] is True
    assert flags["under_20_words"] is True


def test_deterministic_flags_regular_report() -> None:
    flags = deterministic_flags(
        "The heart is mildly enlarged. There is pulmonary vascular congestion "
        "without focal consolidation, pleural effusion, or pneumothorax."
    )
    assert flags["unsupported_temporal_language"] is False
    assert flags["exact_cm_mm_measurement"] is False
    assert flags["empty"] is False

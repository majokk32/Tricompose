"""Protected report-generation interfaces for TriCompose."""

from .protected_cxr import FrozenCXRCase, FrozenCXRSource, load_frozen_cxr_source
from .protected_report_run import GeneratedReport, ReportModelSpec, run_report_generation

__all__ = [
    "FrozenCXRCase",
    "FrozenCXRSource",
    "GeneratedReport",
    "ReportModelSpec",
    "load_frozen_cxr_source",
    "run_report_generation",
]


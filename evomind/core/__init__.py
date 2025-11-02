"""Core utilities exposed for EvoMind SDK users."""

from .profiling import generate_profile_report, summarize_dataframe
from .schema_profiler import profile_dataset
from .task_detector import detect_task_type
from .generic_preprocessing import generic_preprocess
from .data_profiler import profile_health, compute_mutual_information, detect_constant_or_duplicate_columns
from .explain import generate_explanations
from .metrics import (
    classification_metrics,
    fairness_metrics,
    regression_metrics,
)
from .insight_summarizer import summarize_insights

__all__ = [
    "generate_profile_report",
    "summarize_dataframe",
    "profile_dataset",
    "detect_task_type",
    "generic_preprocess",
    "profile_health",
    "compute_mutual_information",
    "detect_constant_or_duplicate_columns",
    "generate_explanations",
    "classification_metrics",
    "regression_metrics",
    "fairness_metrics",
    "summarize_insights",
]

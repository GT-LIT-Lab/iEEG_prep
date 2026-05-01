"""iEEG preprocessing pipeline: config, result, and orchestration."""

from .pipeline import PreprocessingConfig, PreprocessingResult, run_preprocessing_pipeline
from .preprocessing import (
    mark_bad_channels,
    filter_raw,
    find_line_noise_channels,
    high_gamma_envelope_gaussian_bank,
    remove_hg_outliers_pchip,
    normalize_envelope,
)

__all__ = [
    "PreprocessingConfig",
    "PreprocessingResult",
    "run_preprocessing_pipeline",
    "mark_bad_channels",
    "filter_raw",
    "find_line_noise_channels",
    "high_gamma_envelope_gaussian_bank",
    "remove_hg_outliers_pchip",
    "normalize_envelope",
]

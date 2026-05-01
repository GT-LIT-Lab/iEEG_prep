from .utils import add_coordinates, label_channels, rename_channels
from .prep.preprocessing import (
    mark_bad_channels,
    filter_raw,
    find_line_noise_channels,
    high_gamma_envelope_gaussian_bank,
    remove_hg_outliers_pchip,
    normalize_envelope,
)
from .prep import PreprocessingConfig, PreprocessingResult, run_preprocessing_pipeline
from .viz import plot_glass_brain
from .task_analysis import (
    extract_blocks,
    save_block_info,
    load_block,
    run_langloc_pipeline,
    DEFAULT_EVENT_CODES,
    get_trial_word_boundaries_from_block,
    compute_word_starts,
    compute_response_vector,
    permutation_test,
    build_trial_tensor,
    MULTISEM_EVENT_CODES,
    MULTISEM_CONDITION_KEYS,
    get_multisem_trials_from_block,
)

__all__ = [
    # utils
    "add_coordinates",
    "label_channels",
    "rename_channels",
    # preprocessing functions
    "mark_bad_channels",
    "filter_raw",
    "find_line_noise_channels",
    "high_gamma_envelope_gaussian_bank",
    "remove_hg_outliers_pchip",
    "normalize_envelope",
    # preprocessing pipeline
    "PreprocessingConfig",
    "PreprocessingResult",
    "run_preprocessing_pipeline",
    # visualization
    "plot_glass_brain",
    # block extraction
    "extract_blocks",
    "save_block_info",
    "load_block",
    # language localizer
    "run_langloc_pipeline",
    "DEFAULT_EVENT_CODES",
    "get_trial_word_boundaries_from_block",
    "compute_word_starts",
    "compute_response_vector",
    "permutation_test",
    "build_trial_tensor",
    # multisem
    "MULTISEM_EVENT_CODES",
    "MULTISEM_CONDITION_KEYS",
    "get_multisem_trials_from_block",
]

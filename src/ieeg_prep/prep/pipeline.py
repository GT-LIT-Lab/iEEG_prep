"""iEEG preprocessing pipeline: config, result dataclass, and orchestration function."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import mne
import numpy as np

from ..utils import label_channels, add_coordinates, rename_channels
from .preprocessing import (
    mark_bad_channels,
    filter_raw,
    find_line_noise_channels,
    high_gamma_envelope_gaussian_bank,
    remove_hg_outliers_pchip,
    normalize_envelope,
)


@dataclass
class PreprocessingConfig:
    """Full configuration for the iEEG preprocessing pipeline.

    Fields are grouped by pipeline stage. All fields have defaults so any
    subset can be overridden from a JSON config or CLI flags.
    """

    # --- Stage 0: Channel setup ---
    channel_renames: list[dict] = field(default_factory=list)
    """Patient-specific channel name corrections, e.g.
    ``[{"name": "RPuI", "new_name": "RPul"}]``. Empty list skips this step."""
    coordinates_csv: str | None = None
    """Path to MNI electrode coordinates CSV. None skips add_coordinates."""
    coord_channel_col: str = "label"
    coord_x_col: str = "mni_x"
    coord_y_col: str = "mni_y"
    coord_z_col: str = "mni_z"

    # --- Stage 1: Events ---
    stim_channel: str = "TRIG"
    """Channel name passed to mne.find_events(). Events set to None on failure."""

    # --- Stage 2: Resample raw to working frequency ---
    sfreq_working: float = 400.0
    """Raw is resampled to this frequency before all filtering and HG steps."""

    # --- Stage 3: Line noise detection ---
    line_freq: float = 60.0
    """Base line frequency in Hz (60 US, 50 Europe)."""
    max_line_freq: float = 180.0
    line_noise_peak_half_width: float = 2.0
    line_noise_bg_inner_gap: float = 3.0
    line_noise_bg_outer_width: float = 10.0
    line_noise_threshold_sd: float = 5.0
    line_noise_n_fft: int = 4096

    # --- Stage 4: IED-based bad channels ---
    ied_results_path: str | None = None
    """Path to ied_results.npz from a prior IED pipeline run. The CLI loads
    ``channels_above_rate`` from this file and passes it as ``ied_mask``."""

    # --- Stage 5: Filtering ---
    filter_l_freq: float = 0.5
    filter_h_freq: float | None = None
    notch_freqs: list[float] = field(default_factory=lambda: [60.0, 120.0, 180.0])
    notch_method: str = "iir"

    # --- Stage 6: High-gamma envelope ---
    hg_center_freqs: tuple[float, ...] = (73.0, 79.5, 87.8, 96.9, 107.0, 118.1, 130.4, 144.0)
    hg_sigmas: tuple[float, ...] = (4.68, 4.92, 5.17, 5.43, 5.7, 5.99, 6.3, 6.62)
    hg_output: str = "mean"
    """``"mean"`` returns one channel per sEEG channel; ``"all"`` returns one
    per (channel × filter)."""
    hg_ch_type_out: str = "misc"

    # --- Stage 7: Envelope outlier removal ---
    outlier_percentile: float = 90.0
    outlier_multiplier: float = 5.0

    # --- Stage 8: Envelope downsample ---
    hg_downsample_factor: int = 4
    """Downsample factor applied to the HG envelope after outlier removal.
    New sfreq = sfreq_working / hg_downsample_factor (e.g. 400 / 4 = 100 Hz)."""

    # --- Stage 9: Normalization + final resample ---
    normalize_zscore: bool = True
    normalize_minmax: bool = True
    target_sfreq: float = 60.0
    """Final sampling frequency of the saved envelope."""


@dataclass
class PreprocessingResult:
    """Output of :func:`run_preprocessing_pipeline`.

    Core fields are always populated. Intermediate fields are None unless
    ``return_intermediates=True`` was passed to the pipeline.
    """

    # --- Always present ---
    env_norm_raw: mne.io.RawArray
    """Normalized, resampled high-gamma envelope. Ready to save as .fif.
    Channel names are ``{seeg_ch}_hg``. Bad channels propagated from sEEG."""
    events: np.ndarray | None
    """MNE events array resampled to target_sfreq. None if no events found."""
    ch_names_seeg: list[str]
    """Ordered sEEG channel names aligned to the envelope channels."""
    bad_channels: list[str]
    """Union of line-noise and IED bad channels (sEEG names)."""
    sfreq_final: float
    """Actual final sampling frequency of env_norm_raw."""

    # --- Intermediates (None unless return_intermediates=True) ---
    raw_filtered: mne.io.Raw | None = None
    """Post-filter, post-CAR raw at sfreq_working."""
    hg_raw: mne.io.RawArray | None = None
    """High-gamma envelope before outlier removal."""
    hg_clean_raw: mne.io.RawArray | None = None
    """High-gamma envelope after PCHIP outlier removal, before downsample."""
    line_noise_scores: dict[str, float] | None = None
    """Per-channel harmonic noise scores from find_line_noise_channels."""
    outlier_mask: np.ndarray | None = None
    """Boolean array (n_channels, n_times) of detected envelope outliers."""


def run_preprocessing_pipeline(
    raw: mne.io.Raw,
    config: PreprocessingConfig | None = None,
    ied_mask: np.ndarray | None = None,
    return_intermediates: bool = False,
) -> PreprocessingResult:
    """Run the full iEEG preprocessing pipeline.

    Stages in order:

    0. Rename + label channels, add MNI coordinates
    1. Find events
    2. Resample raw to working frequency
    3. Detect line-noise channels
    4. Mark bad channels (line noise + IED mask)
    5. High-pass + notch filter
    6. Common average reference
    7. High-gamma envelope (Gaussian bank + Hilbert)
    8. PCHIP outlier removal
    9. Downsample envelope by ``hg_downsample_factor``
    10. Normalize (z-score and/or min-max) + resample envelope to target_sfreq
    11. Propagate bad channels to envelope

    Args:
        raw: Loaded MNE Raw object (e.g. from mne.io.read_raw_edf). Copied
            internally so the caller's object is not modified.
        config: Pipeline configuration. Uses all defaults if None.
        ied_mask: Boolean array of length equal to the number of sEEG channels,
            marking IED-active channels as bad. None skips IED bad marking.
            Load from ``ied_results.npz["channels_above_rate"]``.
        return_intermediates: If True, populate the intermediate fields of
            :class:`PreprocessingResult` (increases memory usage significantly).

    Returns:
        PreprocessingResult with the normalized envelope and metadata.
    """
    if config is None:
        config = PreprocessingConfig()

    raw = raw.copy()

    # --- Stage 0: Channel setup ---
    if config.channel_renames:
        raw = rename_channels(raw, config.channel_renames)

    raw = label_channels(raw)

    if config.coordinates_csv is not None:
        raw = add_coordinates(
            raw,
            csv_path=config.coordinates_csv,
            channel_col=config.coord_channel_col,
            x_col=config.coord_x_col,
            y_col=config.coord_y_col,
            z_col=config.coord_z_col,
        )

    # --- Stage 1: Events ---
    events_out: np.ndarray | None = None
    try:
        events = mne.find_events(raw, stim_channel=config.stim_channel, verbose=False)
        _, events_out = raw.copy().resample(config.target_sfreq, events=events)
    except (ValueError, RuntimeError):
        events = None

    # --- Stage 2: Resample to working frequency ---
    raw = raw.resample(config.sfreq_working)

    # --- Stage 3: Line noise detection ---
    line_noise_scores, line_noise_bads, _ = find_line_noise_channels(
        raw,
        line_freq=config.line_freq,
        max_freq=config.max_line_freq,
        peak_half_width=config.line_noise_peak_half_width,
        bg_inner_gap=config.line_noise_bg_inner_gap,
        bg_outer_width=config.line_noise_bg_outer_width,
        threshold_sd=config.line_noise_threshold_sd,
        n_fft=config.line_noise_n_fft,
    )

    # --- Stage 4: Mark bad channels ---
    raw = mark_bad_channels(raw, ied_mask=ied_mask, line_noise_bads=line_noise_bads)
    bad_channels = list(raw.info["bads"])

    # --- Stage 5: Filter ---
    raw = filter_raw(
        raw,
        l_freq=config.filter_l_freq,
        h_freq=config.filter_h_freq,
        notch_freqs=config.notch_freqs,
        notch_method=config.notch_method,
    )

    # --- Stage 6: Common average reference ---
    raw_car = raw.copy().set_eeg_reference(ref_channels="average", ch_type="seeg")
    ch_names_seeg: list[str] = raw_car.copy().pick("seeg").ch_names

    # --- Stage 7: High-gamma envelope ---
    hg_raw, _ = high_gamma_envelope_gaussian_bank(
        raw_car,
        center_freqs=config.hg_center_freqs,
        sigmas=config.hg_sigmas,
        output=config.hg_output,
        ch_type_out=config.hg_ch_type_out,
    )

    # --- Stage 8: PCHIP outlier removal ---
    hg_clean_raw, outlier_mask, _ = remove_hg_outliers_pchip(
        hg_raw,
        percentile=config.outlier_percentile,
        multiplier=config.outlier_multiplier,
        verbose=False,
    )

    # --- Stage 9: Downsample envelope ---
    env_sfreq = config.sfreq_working / config.hg_downsample_factor
    env_raw = hg_clean_raw.copy().resample(env_sfreq)

    # --- Stage 10: Normalize + final resample ---
    env_norm_raw = normalize_envelope(
        env_raw,
        zscore=config.normalize_zscore,
        minmax=config.normalize_minmax,
        target_sfreq=config.target_sfreq,
    )

    # --- Stage 11: Propagate bad channels to envelope ---
    seeg_bads = set(bad_channels)
    env_norm_raw.info["bads"] = [
        ch for ch in env_norm_raw.ch_names
        if ch.split("_hg")[0] in seeg_bads
    ]

    result = PreprocessingResult(
        env_norm_raw=env_norm_raw,
        events=events_out,
        ch_names_seeg=ch_names_seeg,
        bad_channels=bad_channels,
        sfreq_final=env_norm_raw.info["sfreq"],
        raw_filtered=raw_car if return_intermediates else None,
        hg_raw=hg_raw if return_intermediates else None,
        hg_clean_raw=hg_clean_raw if return_intermediates else None,
        line_noise_scores=line_noise_scores if return_intermediates else None,
        outlier_mask=outlier_mask if return_intermediates else None,
    )

    if not return_intermediates:
        del raw, raw_car, hg_raw, hg_clean_raw, env_raw
        gc.collect()

    return result

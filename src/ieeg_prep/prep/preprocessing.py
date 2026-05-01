"""iEEG preprocessing pipeline: filtering, bad-channel detection, high-gamma envelope."""

from __future__ import annotations

import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert
from scipy.interpolate import PchipInterpolator


def mark_bad_channels(
    raw: mne.io.BaseRaw,
    ied_mask: np.ndarray | None = None,
    line_noise_bads: list[str] | None = None,
    picks: str = "seeg",
) -> mne.io.BaseRaw:
    """Add bad channels to raw.info["bads"] from IED and/or line-noise sources.

    Modifies the Raw object in place and returns it. Duplicates are ignored.

    Args:
        raw: MNE Raw object.
        ied_mask: Boolean array of length equal to the number of ``picks``
            channels. Channels where the mask is True are marked bad.
        line_noise_bads: List of channel names flagged by
            :func:`find_line_noise_channels` (or any other source).
        picks: Channel type used to resolve ``ied_mask`` indices.

    Returns:
        The same Raw object with ``raw.info["bads"]`` updated.
    """
    if line_noise_bads is not None:
        new = [ch for ch in line_noise_bads if ch not in raw.info["bads"]]
        raw.info["bads"].extend(new)

    if ied_mask is not None:
        ch_names = np.array(raw.copy().pick(picks).ch_names)
        ied_bads = ch_names[ied_mask].tolist()
        new = [ch for ch in ied_bads if ch not in raw.info["bads"]]
        raw.info["bads"].extend(new)

    return raw


def filter_raw(
    raw: mne.io.BaseRaw,
    picks: str = "seeg",
    l_freq: float = 0.5,
    h_freq: float | None = None,
    notch_freqs: list[float] | None = None,
    notch_method: str = "iir",
) -> mne.io.BaseRaw:
    """Apply high-pass and notch filters to a Raw object in place.

    Args:
        raw: MNE Raw object. Modified in place.
        picks: Channels to filter.
        l_freq: High-pass cutoff in Hz. Set to None to skip high-pass.
        h_freq: Low-pass cutoff in Hz. None means no low-pass.
        notch_freqs: Frequencies to notch out (e.g. ``[60, 120, 180]``).
            None skips notch filtering.
        notch_method: Filter method passed to :func:`mne.io.Raw.notch_filter`.

    Returns:
        The same Raw object after filtering.
    """
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, phase="zero")

    if notch_freqs is not None:
        for f in notch_freqs:
            raw.notch_filter(freqs=f, method=notch_method, picks=picks)

    return raw


def find_line_noise_channels(
    raw: mne.io.BaseRaw,
    picks: str = "seeg",
    line_freq: float = 60.0,
    max_freq: float = 180.0,
    peak_half_width: float = 2.0,
    bg_inner_gap: float = 3.0,
    bg_outer_width: float = 10.0,
    threshold_sd: float = 5.0,
    n_fft: int = 4096,
) -> tuple[dict[str, float], list[str], float]:
    """Identify channels with excessive line noise via harmonic power ratio.

    For each harmonic of ``line_freq`` up to ``max_freq``, computes the ratio
    of power in the narrow peak band to power in flanking background bands.
    Log-summed scores across harmonics are thresholded at
    ``mean + threshold_sd * std``.

    Args:
        raw: MNE Raw object.
        picks: Channels to evaluate.
        line_freq: Base line frequency in Hz (60 in US, 50 in Europe).
        max_freq: Highest harmonic to include.
        peak_half_width: Half-width (Hz) around each harmonic for the noise peak.
        bg_inner_gap: Gap (Hz) between peak and background bands.
        bg_outer_width: Width (Hz) of each background sideband.
        threshold_sd: Channels above ``mean + threshold_sd * std`` are flagged.
        n_fft: Welch FFT length.

    Returns:
        scores: Per-channel harmonic noise score.
        bads: Channel names exceeding the threshold.
        threshold: Detection threshold value.
    """
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    ch_names = np.array(raw.copy().pick(picks).ch_names)

    psds, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=max_freq + bg_outer_width + 5,
        n_fft=n_fft,
        average="mean",
        verbose=False,
    )
    psds = np.maximum(psds, np.finfo(float).eps)

    harmonics = np.arange(line_freq, max_freq + 0.1, line_freq)
    total_score = np.zeros(len(ch_names), dtype=float)

    for h in harmonics:
        peak_band = (freqs >= h - peak_half_width) & (freqs <= h + peak_half_width)
        left_bg = (freqs >= h - bg_inner_gap - bg_outer_width) & (freqs < h - bg_inner_gap)
        right_bg = (freqs > h + bg_inner_gap) & (freqs <= h + bg_inner_gap + bg_outer_width)
        bg_band = left_bg | right_bg

        if not peak_band.any() or not bg_band.any():
            continue

        peak_power = psds[:, peak_band].mean(axis=1)
        bg_power = psds[:, bg_band].mean(axis=1)
        total_score += np.log10(peak_power / bg_power)

    mean_score = total_score.mean()
    std_score = total_score.std()
    threshold = mean_score + threshold_sd * std_score

    bad_mask = total_score > threshold
    bads = ch_names[bad_mask].tolist()
    scores = dict(zip(ch_names, total_score))

    return scores, bads, threshold


def high_gamma_envelope_gaussian_bank(
    raw: mne.io.BaseRaw,
    picks: str = "seeg",
    center_freqs: tuple[float, ...] = (73, 79.5, 87.8, 96.9, 107, 118.1, 130.4, 144),
    sigmas: tuple[float, ...] = (4.68, 4.92, 5.17, 5.43, 5.7, 5.99, 6.3, 6.62),
    output: str = "mean",
    ch_type_out: str = "misc",
) -> tuple[mne.io.RawArray, np.ndarray]:
    """Compute high-gamma envelope using a Gaussian filter bank and Hilbert transform.

    For each center frequency, applies a Gaussian filter in the frequency domain,
    inverse-FFTs back to time domain, then takes the absolute value of the analytic
    signal. Returns the mean (or per-filter) envelope across the bank.

    Args:
        raw: Input Raw object.
        picks: Channels to process.
        center_freqs: Gaussian filter center frequencies in Hz.
        sigmas: Gaussian standard deviations in Hz, same length as ``center_freqs``.
        output: ``"mean"`` returns one channel per input channel (averaged across
            filters); ``"all"`` returns one channel per (input channel × filter).
        ch_type_out: Channel type assigned to the output RawArray.

    Returns:
        hg_raw: RawArray containing the high-gamma envelope(s).
        envelopes: Array of shape ``(n_channels, n_filters, n_times)`` with
            per-filter envelopes before averaging.

    Raises:
        ValueError: If ``center_freqs`` and ``sigmas`` differ in length, or
            ``output`` is not ``"mean"`` or ``"all"``.
    """
    center_freqs = np.asarray(center_freqs, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if len(center_freqs) != len(sigmas):
        raise ValueError("center_freqs and sigmas must have the same length.")

    raw_pick = raw.copy().pick(picks)
    data = raw_pick.get_data()
    sfreq = raw_pick.info["sfreq"]
    ch_names = raw_pick.ch_names

    n_channels, n_times = data.shape
    n_filters = len(center_freqs)

    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    data_fft = np.fft.rfft(data, axis=1)

    envelopes = np.empty((n_channels, n_filters, n_times), dtype=np.float64)

    for k, (f0, sigma) in enumerate(zip(center_freqs, sigmas)):
        gaussian = np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)
        filtered = np.fft.irfft(data_fft * gaussian[None, :], n=n_times, axis=1)
        envelopes[:, k, :] = np.abs(hilbert(filtered, axis=1))

    if output == "mean":
        hg_data = envelopes.mean(axis=1)
        out_ch_names = [f"{ch}_hg" for ch in ch_names]
    elif output == "all":
        hg_data = envelopes.reshape(n_channels * n_filters, n_times)
        out_ch_names = [f"{ch}_hg_{cf:g}" for ch in ch_names for cf in center_freqs]
    else:
        raise ValueError("output must be 'mean' or 'all'.")

    info_out = mne.create_info(
        ch_names=out_ch_names,
        sfreq=sfreq,
        ch_types=[ch_type_out] * len(out_ch_names),
    )
    return mne.io.RawArray(hg_data, info_out), envelopes


def remove_hg_outliers_pchip(
    hg_raw: mne.io.BaseRaw,
    picks: str | list | None = None,
    percentile: float = 90.0,
    multiplier: float = 5.0,
    output_dtype: np.dtype = np.float32,
    verbose: bool = True,
) -> tuple[mne.io.RawArray, np.ndarray, np.ndarray]:
    """Detect and replace envelope outliers channel-wise using PCHIP interpolation.

    A sample is an outlier if it exceeds ``multiplier * percentile(channel)``.
    Outlier samples are replaced by interpolating from the surrounding clean samples.

    Args:
        hg_raw: Raw object containing high-gamma envelope data.
        picks: Channels to process. None processes all channels.
        percentile: Percentile defining the baseline high value (default 90).
        multiplier: Threshold multiplier applied to that percentile (default 5.0).
        output_dtype: Dtype for the cleaned output array.
        verbose: Print per-channel replacement summary.

    Returns:
        hg_clean_raw: Cleaned RawArray with the same channels and sfreq as input.
        outlier_mask: Boolean array of shape ``(n_channels, n_times)``.
        thresholds: Per-channel detection thresholds, shape ``(n_channels,)``.
    """
    raw_in = hg_raw.copy()
    if picks is not None:
        raw_in.pick(picks)

    data = raw_in.get_data().astype(np.float64, copy=True)
    ch_names = raw_in.ch_names
    n_channels, n_times = data.shape

    thresholds = multiplier * np.percentile(data, percentile, axis=1)
    outlier_mask = data > thresholds[:, None]

    x = np.arange(n_times)
    for ch in range(n_channels):
        bad = outlier_mask[ch]
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            if verbose:
                print(f"{ch_names[ch]}: skipped (fewer than 2 non-outlier points)")
            continue
        data[ch, bad] = PchipInterpolator(x[good], data[ch, good], extrapolate=True)(x[bad])
        if verbose:
            print(f"{ch_names[ch]}: replaced {bad.sum()} samples ({100 * bad.mean():.4f}%)")

    hg_clean_raw = mne.io.RawArray(data.astype(output_dtype, copy=False), raw_in.info.copy())
    return hg_clean_raw, outlier_mask, thresholds


def normalize_envelope(
    env_raw: mne.io.BaseRaw,
    zscore: bool = True,
    minmax: bool = True,
    target_sfreq: float | None = 60.0,
) -> mne.io.RawArray:
    """Normalize high-gamma envelope data channel-wise and optionally resample.

    Normalization is applied in this order when both flags are True:
    z-score first, then min-max. Each step handles zero variance/range
    gracefully by leaving those channels unchanged.

    Args:
        env_raw: Raw object containing envelope data.
        zscore: Apply z-score normalization (zero mean, unit variance).
        minmax: Apply min-max normalization to the [0, 1] range.
        target_sfreq: Resample to this frequency after normalization.
            Pass None to skip resampling.

    Returns:
        Normalized (and optionally resampled) RawArray.
    """
    data = env_raw.get_data().astype(np.float64, copy=True)

    if zscore:
        means = data.mean(axis=1, keepdims=True)
        stds = data.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        data = (data - means) / stds

    if minmax:
        mins = data.min(axis=1, keepdims=True)
        maxs = data.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        data = (data - mins) / ranges

    norm_raw = mne.io.RawArray(data.astype(np.float32, copy=False), env_raw.info.copy())

    if target_sfreq is not None:
        norm_raw = norm_raw.resample(target_sfreq)

    return norm_raw

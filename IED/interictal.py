"""
Interictal pipeline: filter → envelope → lognormal threshold → spike detection. Adapted from Janca et al. (2015) [https://link.springer.com/article/10.1007/s10548-014-0379-1]

Processes MNE Raw (sEEG) to produce a binary spike marker and per-channel rates.
Modular steps; no file I/O (caller loads/saves).

Array shapes are annotated with jaxtyping (Float/Bool/Int[Array, "dim1 dim2"]) for
documentation and optional static checking.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Literal

import mne
import numpy as np
from jaxtyping import Bool, Float, Int

# Type alias for shaped arrays (numpy; use jax.Array if running under JAX)
Array = np.ndarray
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.special import erf


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class InterictalConfig:
    """Parameters for the interictal (IED) pipeline. All frequencies in Hz."""

    # Resampling
    sfreq_target: float = 200.0

    # Bandpass (applied to sEEG)
    bandpass_low: float = 10.0
    bandpass_high: float = 60.0
    bandpass_order: int = 8
    bandpass_ftype: str = "cheby2"
    bandpass_rs: float = 60.0

    # Notch (line noise)
    notch_freq: float = 60.0
    notch_radius: float = 0.985  # pole radius for custom biquad
    notch_method: Literal["custom_biquad", "mne"] = "custom_biquad"

    # Envelope / lognormal
    win_coeff: float = 5.0  # window length (#samples) = win_coeff (seconds) * sfreq (Hz)
    overlap_coeff: float = 4.0  # overlap (#samples) = overlap_coeff (seconds) * sfreq (Hz)
    smooth_window_len: int = 5  # moving average length for mu/sigma

    # Threshold
    threshold_k: float = 3.65

    # Detection
    polyspike_union_sec: float = 0.12 #Spikes within this time frame are merged into a single spike
    trim_edges_sec: float = 1  # drop first/last N seconds (1 second in this case)

    def bandpass_iir_params(self) -> dict:
        return dict(
            order=self.bandpass_order,
            ftype=self.bandpass_ftype,
            rs=self.bandpass_rs,
        )


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------


def bandpass_raw(
    raw: mne.io.Raw,
    l_freq: float,
    h_freq: float,
    iir_params: dict | None = None,
    picks: str | list = "seeg",
) -> mne.io.Raw:
    """Bandpass filter Raw Default IIR: Cheby2 order 8."""
    if iir_params is None:
        iir_params = InterictalConfig().bandpass_iir_params()
    out = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        picks=picks,
    )
    return out


def notch_custom_biquad(
    raw: mne.io.Raw,
    f0: float = 60.0,
    r: float = 0.985,
    picks: str | list = "seeg",
) -> mne.io.Raw:
    """Apply paper-style biquad notch to sEEG channels only; other channels unchanged.
    
    Janca et al. (2015) used this filter to remove 50 Hz line noise from the data."""
    fs = raw.info["sfreq"]
    w0 = 2 * np.pi * f0 / fs
    b = np.array([1.0, -2 * np.cos(w0), 1.0])
    a = np.array([1.0, -2 * r * np.cos(w0), r**2])
    _, h0 = signal.freqz(b, a, worN=[0], fs=fs)
    b = b / np.abs(h0[0])

    seeg_picks = mne.pick_types(raw.info, seeg=True) if picks == "seeg" else np.atleast_1d(picks)
    if seeg_picks.size == 0:
        return raw.copy()
    data = raw.get_data()
    data_filt = data.copy()
    data_filt[seeg_picks, :] = signal.filtfilt(b, a, data[seeg_picks, :], axis=-1)

    out = raw.copy()
    out._data = data_filt
    return out


def notch_mne(
    raw: mne.io.Raw,
    freqs: list[float] | None = None,
    picks: str | list = "seeg",
) -> mne.io.Raw:
    """Apply MNE IIR notch filter (copy)."""
    if freqs is None:
        freqs = [60.0]
    return raw.copy().notch_filter(freqs=freqs, method="iir", picks=picks)


# -----------------------------------------------------------------------------
# Envelope & lognormal
# -----------------------------------------------------------------------------


def compute_envelope(raw: mne.io.Raw, picks: str | list = "seeg") -> mne.io.Raw:
    """Hilbert envelope of Raw (copy)."""
    return raw.copy().apply_hilbert(picks=picks, envelope=True)


def fit_lognormal_envelope_segments(
    envelope: Float[Array, "n_channels n_times"],
    sfreq: float,
    win_coeff: float = 5.0,
    overlap_coeff: float = 4.0,
) -> tuple[
    Float[Array, "n_channels n_segments"],
    Float[Array, "n_channels n_segments"],
    Int[Array, "n_segments"],
]:
    """
    MLE lognormal (mu, sigma) per segment for each channel.

    Returns mu_hat, sigma_hat (n_channels, n_segments), index (n_segments,).
    """
    winsize = int(win_coeff * sfreq)
    overlap = int(overlap_coeff * sfreq)
    step = max(1, winsize - overlap)
    n_times = envelope.shape[1]
    index = np.arange(0, n_times, step, dtype=int)
    index = index[index + winsize <= n_times]

    if index.size == 0:
        raise ValueError("No valid segments: winsize or step too large for data length.")

    all_windows = sliding_window_view(envelope, window_shape=winsize, axis=1)
    segments = all_windows[:, index, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_segments = np.where(segments > 0, np.log(segments), np.nan)
    mu_hat = np.nanmean(log_segments, axis=-1)
    sigma_hat = np.nanstd(log_segments, axis=-1, ddof=0)
    return mu_hat, sigma_hat, index


def smooth_segment_params(
    mu_hat: Float[Array, "n_channels n_segments"],
    sigma_hat: Float[Array, "n_channels n_segments"],
    window_len: int = 5,
) -> tuple[
    Float[Array, "n_channels n_segments"],
    Float[Array, "n_channels n_segments"],
]:
    """Moving average along segment axis (axis=1)."""
    kernel = np.ones(window_len) / window_len
    mu_smooth = signal.filtfilt(kernel, [1], mu_hat, axis=1)
    sigma_smooth = signal.filtfilt(kernel, [1], sigma_hat, axis=1)
    return mu_smooth, sigma_smooth


def interpolate_phat(
    phat: Float[Array, "n_channels n_segments"],
    index: Int[Array, "n_segments"],
    winsize: int,
    n_samples: int,
) -> Float[Array, "n_channels n_samples"]:
    """
    Interpolate segment-wise phat (n_channels, n_segments) to (n_channels, n_samples).
    """
    half_win = round(winsize / 2)
    centers = index + half_win
    query = np.arange(centers[0], min(centers[-1] + 1, n_samples))
    if query.size == 0:
        return np.broadcast_to(phat[:, :1], (phat.shape[0], n_samples)).copy()
    cs = CubicSpline(centers, phat.T)
    interp = cs(query).T
    n_start = half_win
    n_end = n_samples - (interp.shape[1] + n_start)
    pad_start = np.broadcast_to(interp[:, :1], (interp.shape[0], n_start)).copy()
    pad_end = np.broadcast_to(interp[:, -1:], (interp.shape[0], n_end)).copy()
    return np.concatenate([pad_start, interp, pad_end], axis=1)


def compute_threshold(
    mu_interp: Float[Array, "n_channels n_samples"],
    sigma_interp: Float[Array, "n_channels n_samples"],
    k: float = 3.65,
) -> Float[Array, "n_channels n_samples"]:
    """Threshold = k * (logn_mode + logn_median)."""
    logn_mode = np.exp(mu_interp - sigma_interp**2)
    logn_median = np.exp(mu_interp)
    return k * (logn_mode + logn_median)


# -----------------------------------------------------------------------------
# Detection: crossings → spike peaks → polyspike union → single max → trim
# -----------------------------------------------------------------------------


def _crossings(
    marker: Float[Array, "n_channels n_times"],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """(n_channels, n_times) -> start/end (each tuple of (ch_idx, sample_idx))."""
    pad_l = np.pad(marker, ((0, 0), (1, 0)))
    pad_r = np.pad(marker, ((0, 0), (0, 1)))
    start_crossing = np.where(np.diff(pad_l, axis=1) > 0)
    end_crossing = np.where(np.diff(pad_r, axis=1) < 0)
    return start_crossing, end_crossing


def _peaks_in_crossings(
    envelope: Float[Array, "n_channels n_samples"],
    start_crossing: tuple[np.ndarray, np.ndarray],
    end_crossing: tuple[np.ndarray, np.ndarray],
    n_channels: int,
    n_samples: int,
) -> Bool[Array, "n_channels n_samples"]:
    """One peak per (channel, crossing)."""
    spike_markers = np.zeros((n_channels, n_samples), dtype=bool)
    n_events = start_crossing[0].shape[0]
    for i in range(n_events):
        ch = start_crossing[0][i]
        start = start_crossing[1][i]
        end = end_crossing[1][i]
        seg = envelope[ch, start : end + 1]
        # If the segment is longer than 2 samples, find the peaks using the sign of the difference
        if end - start > 2:
            seg_diff = np.sign(np.diff(seg))
            changes = np.diff(np.concatenate(([0], seg_diff)))
            peak_offsets = np.where(changes < 0)[0]
            for off in peak_offsets:
                spike_markers[ch, start + off] = True
        else:
            s_max = int(np.argmax(seg))
            spike_markers[ch, start + s_max] = True
    return spike_markers


def _polyspike_union(
    spike_markers: Bool[Array, "n_channels n_samples"],
    sfreq: float,
    union_sec: float,
) -> Bool[Array, "n_channels n_samples"]:
    """Merge spike markers within union_sec (per channel)."""
    out = spike_markers.copy()
    sample_wind = int(sfreq * union_sec)
    n_channels, n_samples = out.shape
    spike_ch, spike_t = np.where(out)
    # Process per channel
    for ch in range(n_channels):
        times = spike_t[spike_ch == ch]
        if times.size == 0:
            continue
        times = np.sort(times)
        i = 0
        while i < len(times):
            start = times[i]
            j = i + 1
            while j < len(times) and times[j] - start <= sample_wind:
                j += 1
            end = times[j - 1]
            out[ch, start : end + 1] = True
            i = j
    return out


def _local_max_per_crossing(
    spike_markers: Bool[Array, "n_channels n_samples"],
    envelope: Float[Array, "n_channels n_samples"],
) -> Bool[Array, "n_channels n_samples"]:
    """Replace each crossing run with a single peak at the local max of envelope."""
    start_crossing, end_crossing = _crossings(spike_markers.astype(np.float64))
    out = np.zeros_like(spike_markers, dtype=bool)
    n_events = start_crossing[0].shape[0]
    spike_ch, spike_t = np.where(spike_markers)
    for k in range(n_events):
        channel = start_crossing[0][k]
        start = start_crossing[1][k]
        end = end_crossing[1][k]
        ch_mask = spike_ch == channel
        channel_pointers = spike_t[ch_mask]
        local_max_mask = (channel_pointers >= start) & (channel_pointers <= end)
        local_max_indices = channel_pointers[local_max_mask]
        if local_max_indices.size == 0:
            continue
        local_max_vals = envelope[channel, local_max_indices]
        # local max: second derivative < 0
        padded = np.concatenate(([0], local_max_vals, [0]))
        second_d = np.diff(np.sign(np.diff(padded)))
        local_max_pos = np.where(second_d < 0)[0]
        if local_max_pos.size > 0:
            best = local_max_pos[np.argmax(local_max_vals[local_max_pos])]
            out[channel, local_max_indices[best]] = True
        else:
            best = int(np.argmax(local_max_vals))
            out[channel, local_max_indices[best]] = True
    return out


def _trim_edges(
    spike_markers: Bool[Array, "n_channels n_samples"],
    sfreq: float,
    trim_sec: float,
) -> Bool[Array, "n_channels n_samples"]:
    """Trim the edges of the spike markers (drop first/last trim_sec seconds)."""
    n = int(sfreq * trim_sec)
    out = spike_markers.copy()
    out[:, :n] = False
    out[:, -n:] = False
    return out


def run_detection(
    envelope: Float[Array, "n_channels n_samples"],
    threshold: Float[Array, "n_channels n_samples"],
    sfreq: float,
    polyspike_union_sec: float = 0.12,
    trim_edges_sec: float = 1.0,
) -> Bool[Array, "n_channels n_samples"]:
    """
    Binary marker (envelope > threshold) → crossings → peaks → polyspike union
    → one local max per event → trim edges.
    """
    n_channels, n_samples = envelope.shape
    marker1 = (envelope > threshold).astype(np.float64)
    start_crossing, end_crossing = _crossings(marker1)
    spike_markers = _peaks_in_crossings(
        envelope, start_crossing, end_crossing, n_channels, n_samples
    )
    spike_markers = _polyspike_union(spike_markers, sfreq, polyspike_union_sec)
    spike_markers = _local_max_per_crossing(spike_markers, envelope)
    spike_markers = _trim_edges(spike_markers, sfreq, trim_edges_sec)
    return spike_markers


# -----------------------------------------------------------------------------
# Results container & pipeline
# -----------------------------------------------------------------------------


@dataclass
class InterictalResult:
    """Output of the interictal pipeline."""

    spike_markers: Bool[Array, "n_channels n_samples"]
    channel_spike_counts: Int[Array, "n_channels"]
    channel_spike_rates: Float[Array, "n_channels"]  # per minute
    sfreq: float = 0.0
    time_min: float = 0.0
    # Optional intermediates (set by pipeline if requested)
    raw_filtered: mne.io.Raw | None = None
    raw_envelope: mne.io.Raw | None = None
    threshold: Float[Array, "n_channels n_samples"] | None = None
    mu_interp: Float[Array, "n_channels n_samples"] | None = None
    sigma_interp: Float[Array, "n_channels n_samples"] | None = None
    envelope_cdf: Float[Array, "n_channels n_samples"] | None = None
    envelope_pdf: Float[Array, "n_channels n_samples"] | None = None

    @property
    def n_spikes_total(self) -> int:
        return int(self.spike_markers.sum())

    def channels_above_rate(self, rate_thresh: float) -> Bool[Array, "n_channels"]:
        """Boolean mask of channels with spike rate > rate_thresh (per minute)."""
        return self.channel_spike_rates > rate_thresh


def run_interictal_pipeline(
    raw: mne.io.Raw,
    config: InterictalConfig | None = None,
    return_intermediates: bool = False,
) -> InterictalResult:
    """
    Run the full interictal pipeline on an MNE Raw (expects sEEG channel types set).

    Steps: resample → bandpass → notch → envelope → lognormal fit → smooth
    → interpolate → threshold → detection (crossings, peaks, polyspike union,
    local max, trim) → spike rates.

    raw : preload=True recommended; channel types should already be set (e.g. label_channels).
    When return_intermediates is False, intermediate Raw objects are deleted after each
    step to limit memory use (multi-GB copies are freed as soon as possible).
    """
    if config is None:
        config = InterictalConfig()

    # Copy and resample
    raw_work = raw.copy()
    if raw_work.info["sfreq"] != config.sfreq_target:
        raw_work.resample(config.sfreq_target)
    sfreq = raw_work.info["sfreq"]

    # Bandpass
    raw_filt = bandpass_raw(
        raw_work,
        config.bandpass_low,
        config.bandpass_high,
        iir_params=config.bandpass_iir_params(),
        picks="seeg",
    )
    del raw_work
    gc.collect()

    # Notch
    if config.notch_method == "custom_biquad":
        raw_notch = notch_custom_biquad(
            raw_filt, f0=config.notch_freq, r=config.notch_radius, picks="seeg"
        )
    else:
        raw_notch = notch_mne(raw_filt, freqs=[config.notch_freq], picks="seeg")
    del raw_filt
    gc.collect()

    # Envelope
    raw_env = compute_envelope(raw_notch, picks="seeg")
    if not return_intermediates:
        del raw_notch
        gc.collect()

    envelope = raw_env.get_data(picks="seeg")
    if not return_intermediates:
        del raw_env
        gc.collect()

    # Lognormal fit
    mu_hat, sigma_hat, index = fit_lognormal_envelope_segments(
        envelope, sfreq, config.win_coeff, config.overlap_coeff
    )
    mu_smooth, sigma_smooth = smooth_segment_params(
        mu_hat, sigma_hat, config.smooth_window_len
    )
    winsize = int(config.win_coeff * sfreq)  # segment window used in lognormal fit
    n_samples = envelope.shape[1]
    mu_interp = interpolate_phat(mu_smooth, index, winsize, n_samples)
    sigma_interp = interpolate_phat(sigma_smooth, index, winsize, n_samples)
    if not return_intermediates:
        del mu_hat, sigma_hat, mu_smooth, sigma_smooth
        gc.collect()

    # Threshold
    threshold = compute_threshold(mu_interp, sigma_interp, config.threshold_k)

    # Detection
    spike_markers = run_detection(
        envelope,
        threshold,
        sfreq,
        polyspike_union_sec=config.polyspike_union_sec,
        trim_edges_sec=config.trim_edges_sec,
    )

    # Rates
    n_channels = envelope.shape[0]
    channel_spike_counts = spike_markers.sum(axis=1)
    time_min = spike_markers.shape[1] / sfreq / 60.0
    channel_spike_rates = np.zeros(n_channels)
    np.divide(channel_spike_counts, time_min, out=channel_spike_rates, where=time_min > 0)

    result = InterictalResult(
        spike_markers=spike_markers,
        channel_spike_counts=channel_spike_counts,
        channel_spike_rates=channel_spike_rates,
        sfreq=sfreq,
        time_min=time_min,
    )
    if return_intermediates:
        result.raw_filtered = raw_notch
        result.raw_envelope = raw_env
        result.threshold = threshold
        result.mu_interp = mu_interp
        result.sigma_interp = sigma_interp
        log_env = np.log(envelope)
        result.envelope_cdf = 0.5 + 0.5 * erf(
            (log_env - mu_interp) / (np.sqrt(2) * sigma_interp)
        )
        result.envelope_pdf = np.exp(
            -0.5 * ((log_env - mu_interp) / sigma_interp) ** 2
        ) / (envelope * sigma_interp * np.sqrt(2 * np.pi))
    else:
        del threshold
        gc.collect()
    return result

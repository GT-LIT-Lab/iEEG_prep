"""Shared localizer statistics: response vector construction and permutation test."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def compute_response_vector(
    pos_trials: list[np.ndarray],
    neg_trials: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a mean-response vector and ideal contrast labels.

    Each trial is summarised as its mean signal across the sample axis,
    so variable-length trial windows are handled naturally.

    Parameters
    ----------
    pos_trials : list of np.ndarray, each shape (n_channels, n_samples)
        Signal arrays for the positive-condition trials.
    neg_trials : list of np.ndarray, each shape (n_channels, n_samples)
        Signal arrays for the negative-condition trials.

    Returns
    -------
    respvec : np.ndarray, shape (n_channels, n_pos + n_neg)
        Mean signal per channel per trial, positive trials first.
    ideal : np.ndarray, shape (n_pos + n_neg,)
        +1 for positive-condition trials, -1 for negative-condition trials.
    """
    n_pos = len(pos_trials)
    n_neg = len(neg_trials)
    n_channels = pos_trials[0].shape[0] if pos_trials else neg_trials[0].shape[0]

    respvec = np.zeros((n_channels, n_pos + n_neg))
    for t, sig in enumerate(pos_trials):
        respvec[:, t] = sig.mean(axis=-1)
    for t, sig in enumerate(neg_trials):
        respvec[:, n_pos + t] = sig.mean(axis=-1)

    ideal = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    return respvec, ideal


def permutation_test(
    respvec: np.ndarray,
    respvec_ideal: np.ndarray,
    n_perm: int = 10000,
    threshold_pct: float = 95.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-sided per-channel Spearman permutation test for localizer responsiveness.

    For each channel, tests whether the observed Spearman correlation with the
    ideal contrast vector exceeds the ``threshold_pct`` percentile of a
    label-shuffled null distribution.

    Parameters
    ----------
    respvec : np.ndarray, shape (n_channels, n_trials)
        Observed responses per channel per trial.
    respvec_ideal : np.ndarray, shape (n_trials,)
        Ideal contrast labels (+1 / -1).
    n_perm : int
        Number of label permutations for the null distribution.
    threshold_pct : float
        Percentile of the null used as the significance threshold (default 95).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    true_corrs : np.ndarray, shape (n_channels,)
        Observed Spearman correlation per channel.
    null_corrs : np.ndarray, shape (n_channels, n_perm)
        Null distribution correlations.
    p_values : np.ndarray, shape (n_channels,)
        One-sided empirical p-values.
    is_responsive : np.ndarray of bool, shape (n_channels,)
        True where ``true_corrs`` exceeds the null threshold.
    thresholds : np.ndarray, shape (n_channels,)
        The ``threshold_pct`` percentile of the null per channel.
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(respvec, dtype=float)
    y = np.asarray(respvec_ideal, dtype=float)

    # Spearman = Pearson on ranks; rank once
    Xr = np.apply_along_axis(rankdata, 1, X)
    yr = rankdata(y)

    Xc = Xr - Xr.mean(axis=1, keepdims=True)
    yc = yr - yr.mean()

    X_norm = np.linalg.norm(Xc, axis=1)
    y_norm = np.linalg.norm(yc)

    true_corrs = (Xc @ yc) / (X_norm * y_norm)

    perms = np.array([rng.permutation(y) for _ in range(n_perm)])
    perms_r = np.apply_along_axis(rankdata, 1, perms)
    perms_c = perms_r - perms_r.mean(axis=1, keepdims=True)
    perms_norm = np.linalg.norm(perms_c, axis=1)

    null_corrs = (Xc @ perms_c.T) / (X_norm[:, None] * perms_norm[None, :])

    thresholds = np.percentile(null_corrs, threshold_pct, axis=1)
    is_responsive = true_corrs > thresholds

    p_values = (np.sum(null_corrs >= true_corrs[:, None], axis=1) + 1) / (n_perm + 1)

    return true_corrs, null_corrs, p_values, is_responsive, thresholds

"""Group-level amplitude statistics for the language localizer."""

from __future__ import annotations

import numpy as np


def amplitude_permutation_test(
    trial_tensor: np.ndarray,
    mask: np.ndarray,
    n_permutations: int = 5000,
    seed: int | None = 42,
) -> dict:
    """One-tailed permutation test: is the group mean (sent - nw) amplitude > 0?

    The observed statistic is the mean across masked electrodes of each
    electrode's (mean sentence amplitude - mean non-word amplitude), where
    amplitude is averaged over both trials and time.  The null distribution is
    built by pooling all trials, shuffling condition labels, and recomputing the
    same statistic.

    Parameters
    ----------
    trial_tensor : np.ndarray, shape (2, n_trials, n_channels, n_time)
        Output of :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.
        Axis 0 must be ``[sentence, non_word]``.
    mask : np.ndarray of bool, shape (n_channels,)
        Selects which electrodes contribute to the group statistic.
    n_permutations : int
        Number of label shuffles for the null distribution.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    result : dict with keys:
        ``observed``      – float, observed group mean (sent - nw) amplitude.
        ``null``          – np.ndarray, shape (n_permutations,), null distribution.
        ``p_value``       – float, one-tailed p-value (fraction of null >= observed).
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            "observed": np.nan,
            "null": np.full(n_permutations, np.nan),
            "p_value": np.nan,
        }
    rng = np.random.default_rng(seed)

    n_cond, n_trials, n_channels, n_time = trial_tensor.shape
    n_total = 2 * n_trials

    # Pool and pre-average over time → (n_total, n_channels)
    # Averaging time first reduces the array before any permutation work
    all_trials = np.concatenate([trial_tensor[0], trial_tensor[1]], axis=0)
    all_trials_avg = all_trials.mean(axis=2)  # (n_total, n_channels)

    # Observed stat
    sent_mean = all_trials_avg[:n_trials, mask].mean(axis=0)   # (n_mask,)
    nw_mean   = all_trials_avg[n_trials:, mask].mean(axis=0)
    observed  = float((sent_mean - nw_mean).mean())

    # Vectorized null: generate all permuted indices at once
    # perm_idx shape: (n_permutations, n_total)
    perm_idx = np.argsort(rng.random((n_permutations, n_total)), axis=1)

    # Select masked channels only before permuting to keep memory down
    masked_avg = all_trials_avg[:, mask]  # (n_total, n_mask)

    # perm_trials: (n_permutations, n_total, n_mask)
    perm_trials = masked_avg[perm_idx]

    # Split into sent / nw halves and compute group-mean difference per permutation
    null = (perm_trials[:, :n_trials, :].mean(axis=1) -
            perm_trials[:, n_trials:, :].mean(axis=1)).mean(axis=1)  # (n_permutations,)

    p_value = float((null >= observed).mean())

    return {"observed": observed, "null": null, "p_value": p_value}

"""MultiSem analysis: trial tensor construction."""

from __future__ import annotations

import numpy as np

from .utils import MULTISEM_CONDITION_KEYS


def build_multisem_trial_tensor(
    trials: list[dict],
    envelope_data: np.ndarray,
    condition_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a 4D trial tensor grouped by condition.

    The epoch for each trial spans from the trial trigger to the ITI trigger
    (i.e. the stimulus presentation period, excluding the inter-trial interval).
    Trials are clipped to the shortest epoch duration across all conditions so
    the output is a regular array.

    Parameters
    ----------
    trials : list of dict
        Output of :func:`~ieeg_prep.task_analysis.multisem.utils.get_multisem_trials_from_block`.
    envelope_data : np.ndarray, shape (n_channels, n_samples)
        High-gamma envelope or other neural data.
    condition_keys : list of str, optional
        Ordered condition names defining axis-0 of the tensor.  Defaults to
        :data:`~ieeg_prep.task_analysis.multisem.utils.MULTISEM_CONDITION_KEYS`
        (the six standard MultiSem conditions).  Conditions absent from
        *trials* raise a :class:`ValueError`.

    Returns
    -------
    tensor : np.ndarray, shape (n_conditions, n_trials, n_channels, min_samples)
        Trial data organised by condition.  ``n_trials`` is the minimum trial
        count across all conditions; ``min_samples`` is the shortest
        stimulus-period duration across all trials.
    conditions : list of str
        Condition names corresponding to tensor axis 0, in the same order as
        *condition_keys*.

    Raises
    ------
    ValueError
        If any condition in *condition_keys* has no valid trials, or if any
        trial produces an empty epoch.
    """
    if condition_keys is None:
        condition_keys = MULTISEM_CONDITION_KEYS

    cond_data: dict[str, list[np.ndarray]] = {k: [] for k in condition_keys}

    for trial in trials:
        label = trial["condition_label"]
        if label not in cond_data:
            continue

        start = trial["trial_start_sample"]
        end   = trial["iti_start_sample"]
        if end <= start:
            continue

        segment = envelope_data[:, start:end]
        if segment.shape[1] == 0:
            continue

        cond_data[label].append(segment)

    for cond in condition_keys:
        if not cond_data[cond]:
            raise ValueError(f"No valid trials found for condition '{cond}'.")

    min_samples = min(
        seg.shape[1]
        for segs in cond_data.values()
        for seg in segs
    )
    n_trials    = min(len(cond_data[c]) for c in condition_keys)
    n_channels  = envelope_data.shape[0]
    n_conditions = len(condition_keys)

    tensor = np.zeros(
        (n_conditions, n_trials, n_channels, min_samples),
        dtype=envelope_data.dtype,
    )

    for c, cond in enumerate(condition_keys):
        for t, seg in enumerate(cond_data[cond][:n_trials]):
            tensor[c, t, :, :] = seg[:, :min_samples]

    return tensor, list(condition_keys)

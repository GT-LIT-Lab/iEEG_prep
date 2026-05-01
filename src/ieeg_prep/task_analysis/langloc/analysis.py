"""Post-langloc analysis: word timing and trial tensor construction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_word_starts(trials: list[dict]) -> np.ndarray:
    """Compute mean word onset times across trials, normalized to trial start.

    Parameters
    ----------
    trials : list of dict
        Output of :func:`~ieeg_prep.task_analysis.langloc.utils.get_trial_word_boundaries_from_block`.
        Each trial must contain a ``word_bounds`` key with ``(start_sample, end_sample)`` pairs.

    Returns
    -------
    word_starts : np.ndarray, shape (n_words,)
        Floor of the mean per-position word onset, in samples relative to the
        first word of each trial.
    """
    word_starts = []
    for trial in trials:
        onsets = np.array([bound[0] for bound in trial["word_bounds"]])
        onsets = onsets - onsets[0]
        word_starts.append(onsets)

    return np.floor(np.mean(word_starts, axis=0)).astype(int)


def build_trial_tensor(
    trials: list[dict],
    envelope_data: np.ndarray,
    event_codes: dict[str, int],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a 4D trial tensor grouped by condition.

    Each trial is clipped to the shortest trial duration across all conditions
    so the output is a regular array.

    Parameters
    ----------
    trials : list of dict
        Output of :func:`~ieeg_prep.task_analysis.langloc.utils.get_trial_word_boundaries_from_block`.
    envelope_data : np.ndarray, shape (n_channels, n_samples)
        High-gamma envelope or other neural data.
    event_codes : dict[str, int]
        Name-to-code mapping used to label conditions (e.g. ``"sentence"``,
        ``"non_word"``).  Any trial whose first word code does not map to a
        name in this dict is skipped.

    Returns
    -------
    tensor : np.ndarray, shape (2, n_trials, n_channels, min_samples)
        Trial data organised by condition.  Axis 0 is always
        ``[sentence, non_word]``.  ``n_trials`` is the minimum trial count
        across the two conditions; ``min_samples`` is the shortest trial
        duration (first word onset to probe) across all trials.
    conditions : list of str
        Always ``["sentence", "non_word"]``.
    word_starts : np.ndarray, shape (n_words,)
        Mean word onset times shared across both conditions (floored, relative
        to each trial's first word onset).

    Raises
    ------
    ValueError
        If either condition has no valid trials.
    """
    CONDITIONS = ["sentence", "non_word"]  # axis-0 order: 0 = sentence, 1 = non-word

    sentence_code = event_codes["sentence"]
    nonword_code = event_codes["non_word"]
    code_to_cond = {sentence_code: "sentence", nonword_code: "non_word"}

    cond_data: dict[str, list[np.ndarray]] = {"sentence": [], "non_word": []}
    cond_trials: dict[str, list[dict]] = {"sentence": [], "non_word": []}

    for trial in trials:
        code = int(trial["word_codes"][0])
        cond = code_to_cond.get(code)
        if cond is None:
            continue

        start = trial["word_bounds"][0][0]
        end = trial["word_bounds"][-1][1]
        if end <= start:
            continue

        segment = envelope_data[:, start:end]
        if segment.shape[1] == 0:
            continue

        cond_data[cond].append(segment)
        cond_trials[cond].append(trial)

    if not cond_data["sentence"]:
        raise ValueError("No valid sentence trials found.")
    if not cond_data["non_word"]:
        raise ValueError("No valid non-word trials found.")

    min_samples = min(
        seg.shape[1]
        for segs in cond_data.values()
        for seg in segs
    )
    n_trials = min(len(cond_data[c]) for c in CONDITIONS)
    n_channels = envelope_data.shape[0]

    tensor = np.zeros(
        (2, n_trials, n_channels, min_samples),
        dtype=envelope_data.dtype,
    )

    all_trials_in_tensor = []
    for c, cond in enumerate(CONDITIONS):
        for t, seg in enumerate(cond_data[cond][:n_trials]):
            tensor[c, t, :, :] = seg[:, :min_samples]
        all_trials_in_tensor.extend(cond_trials[cond][:n_trials])

    onsets = np.array(
        [[bound[0] for bound in trial["word_bounds"]] for trial in all_trials_in_tensor],
        dtype=float,
    )
    onsets -= onsets[:, 0:1]
    word_starts = np.floor(onsets.mean(axis=0)).astype(int)

    return tensor, CONDITIONS, word_starts

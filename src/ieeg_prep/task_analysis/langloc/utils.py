"""Language localizer utilities: trial parsing, response vectors, and permutation test."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# Default name-to-code mapping for the MIT language localizer paradigm.
# Verify these against your own paradigm before use.
DEFAULT_EVENT_CODES: dict[str, int] = {
    "experiment_start": 1,
    "experiment_end": 2,
    "non_word": 3,
    "sentence": 4,
    "trial": 5,
    "fixation": 6,
    "subject_response": 7,
    "white_screen": 8,
    "probe": 9,
}


def get_trial_word_boundaries_from_block(
    block: dict,
    event_codes: dict[str, int] | None = None,
    n_words: int = 12,
) -> tuple[list[dict], list[dict]]:
    """Parse one block dictionary and extract word boundaries for each valid trial.

    Parameters
    ----------
    block : dict
        Block dictionary as returned by :func:`~ieeg_prep.task_analysis.extract_blocks`.
    event_codes : dict[str, int], optional
        Name-to-code mapping. Must contain ``"trial"``, ``"sentence"``,
        ``"non_word"``, and ``"probe"``. Defaults to :data:`DEFAULT_EVENT_CODES`.
    n_words : int
        Expected number of words per trial.

    Returns
    -------
    trials : list of dict
        Valid trials found within the block. Each dict contains:

        - ``trial_number`` (int)
        - ``block_start_event_idx`` (int)
        - ``trial_start_event_idx_local`` (int)
        - ``trial_start_event_idx_global`` (int)
        - ``word_event_indices_local`` (list[int])
        - ``word_event_indices_global`` (list[int])
        - ``word_samples`` (np.ndarray[int])
        - ``word_codes`` (np.ndarray[int])
        - ``probe_event_idx_local`` (int)
        - ``probe_event_idx_global`` (int)
        - ``probe_sample`` (int)
        - ``word_bounds`` (list[tuple[int, int]]): ``(start_sample, end_sample)``
          for each word; the last word ends at the probe.

    bad_trials : list of dict
        Trials with invalid structure (wrong word count or missing probe).
    """
    if event_codes is None:
        event_codes = DEFAULT_EVENT_CODES

    events = block["events"]
    block_start_event_idx = block["start_event_idx"]

    trial_code = event_codes["trial"]
    probe_code = event_codes["probe"]
    word_codes_set = {event_codes["sentence"], event_codes["non_word"]}
    trials: list[dict] = []
    bad_trials: list[dict] = []

    trial_number = 0
    i = 0
    n_events = len(events)

    while i < n_events:
        if events[i, 2] != trial_code:
            i += 1
            continue

        trial_start_idx_local = i
        trial_start_idx_global = block_start_event_idx + i

        word_event_indices_local: list[int] = []
        word_event_indices_global: list[int] = []
        probe_event_idx_local = None
        probe_event_idx_global = None

        j = i + 1
        while j < n_events:
            code = events[j, 2]

            if code == trial_code:
                break

            if code in word_codes_set:
                word_event_indices_local.append(j)
                word_event_indices_global.append(block_start_event_idx + j)

            if code == probe_code:
                probe_event_idx_local = j
                probe_event_idx_global = block_start_event_idx + j
                break

            j += 1

        if probe_event_idx_local is not None and len(word_event_indices_local) == n_words:
            word_samples = events[word_event_indices_local, 0]
            word_trigger_codes = events[word_event_indices_local, 2]
            probe_sample = events[probe_event_idx_local, 0]

            word_bounds: list[tuple[int, int]] = []
            for k in range(n_words - 1):
                word_bounds.append((int(word_samples[k]), int(word_samples[k + 1])))
            word_bounds.append((int(word_samples[-1]), int(probe_sample)))

            trials.append({
                "block_start_event_idx": int(block_start_event_idx),
                "trial_number": trial_number,
                "trial_start_event_idx_local": int(trial_start_idx_local),
                "trial_start_event_idx_global": int(trial_start_idx_global),
                "word_event_indices_local": [int(x) for x in word_event_indices_local],
                "word_event_indices_global": [int(x) for x in word_event_indices_global],
                "word_samples": word_samples.astype(int),
                "word_codes": word_trigger_codes.astype(int),
                "probe_event_idx_local": int(probe_event_idx_local),
                "probe_event_idx_global": int(probe_event_idx_global),
                "probe_sample": int(probe_sample),
                "word_bounds": word_bounds,
            })

            trial_number += 1
            i = probe_event_idx_local + 1

        else:
            bad_trials.append({
                "block_start_event_idx": int(block_start_event_idx),
                "trial_start_event_idx_local": int(trial_start_idx_local),
                "trial_start_event_idx_global": int(trial_start_idx_global),
                "found_n_words": len(word_event_indices_local),
                "found_probe": probe_event_idx_local is not None,
            })
            i = max(i + 1, j)

    return trials, bad_trials


def compute_response_vector(
    trials: list[dict],
    envelope_data: np.ndarray,
    event_codes: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean envelope response per trial and an ideal contrast vector.

    Parameters
    ----------
    trials : list of dict
        Output of :func:`get_trial_word_boundaries_from_block`.
    envelope_data : np.ndarray, shape (n_channels, n_samples)
        High-gamma envelope or other neural data.
    event_codes : dict[str, int]
        Name-to-code mapping. Must contain ``"sentence"`` and ``"non_word"``.

    Returns
    -------
    respvec : np.ndarray, shape (n_channels, n_trials)
        Mean envelope across all word windows, per channel per trial.
    respvec_ideal : np.ndarray, shape (n_trials,)
        +1 for sentence trials, -1 for non-word trials, 0 for other.
    """
    n_channels = envelope_data.shape[0]
    n_trials = len(trials)

    respvec = np.zeros((n_channels, n_trials))
    respvec_ideal = np.zeros(n_trials)

    sentence_code = event_codes["sentence"]
    nonword_code = event_codes["non_word"]

    for t, trial in enumerate(trials):
        stim_resp = np.zeros(n_channels)

        for start_sample, end_sample in trial["word_bounds"]:
            if end_sample <= start_sample:
                print(f"Warning: trial {t} has empty window [{start_sample}, {end_sample}), skipping")
                continue
            if end_sample > envelope_data.shape[1]:
                print(f"Warning: trial {t} end_sample {end_sample} exceeds envelope length, skipping")
                continue
            stim_resp += envelope_data[:, start_sample:end_sample].mean(axis=1)

        respvec[:, t] = stim_resp / len(trial["word_bounds"])

        code = int(trial["word_codes"][0])
        if code == sentence_code:
            respvec_ideal[t] = 1.0
        elif code == nonword_code:
            respvec_ideal[t] = -1.0

    return respvec, respvec_ideal


def permutation_test(
    respvec: np.ndarray,
    respvec_ideal: np.ndarray,
    n_perm: int = 10000,
    threshold_pct: float = 95.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-sided per-channel Spearman permutation test for language responsiveness.

    For each channel, tests whether the observed Spearman correlation with the
    ideal contrast vector exceeds the ``threshold_pct`` percentile of a
    label-shuffled null distribution.

    Parameters
    ----------
    respvec : np.ndarray, shape (n_channels, n_trials)
        Observed responses per channel per trial.
    respvec_ideal : np.ndarray, shape (n_trials,)
        Ideal contrast labels (e.g. +1 for sentence, -1 for non-word).
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
    is_language_responsive : np.ndarray of bool, shape (n_channels,)
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
    is_language_responsive = true_corrs > thresholds

    p_values = (np.sum(null_corrs >= true_corrs[:, None], axis=1) + 1) / (n_perm + 1)

    return true_corrs, null_corrs, p_values, is_language_responsive, thresholds


def load_lang_mask(
    results_path,
    block: str = "superset",
    exclude_bad: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Load a language-responsive mask from a langloc results NPZ file.

    The mask is aligned to the envelope channel order used during the pipeline
    run, so it can be passed directly to
    :func:`~ieeg_prep.viz.langloc_plots.plot_sent_nw` or used to index into
    the trial tensor produced by
    :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.

    Parameters
    ----------
    results_path : str or Path
        Path to the NPZ file written by :func:`~ieeg_prep.task_analysis.langloc.pipeline.run_langloc_pipeline`.
    block : str
        Which result to load.  Use ``"superset"`` (default) for the pooled
        estimate, or a block label such as ``"langloc1"`` for a per-block mask.
    exclude_bad : bool
        If True (default), bad channels are removed from both the mask and
        ``ch_names``, reducing the returned dimensionality.  If False, the
        full channel list is returned with bad channels included.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_good_channels,) or (n_channels,)
        Language-responsive mask.  When ``exclude_bad=True`` bad channels are
        dropped entirely; when False the full pipeline channel order is kept.
    ch_names : list of str
        Channel names in the same order as ``mask``.

    Raises
    ------
    KeyError
        If ``block`` is not found in the results file.
    """
    from pathlib import Path
    data = np.load(Path(results_path), allow_pickle=True)

    key = "superset_is_language_responsive" if block == "superset" else f"{block}_is_language_responsive"
    if key not in data:
        available = [
            k.replace("_is_language_responsive", "")
            for k in data.files
            if k.endswith("_is_language_responsive")
        ]
        raise KeyError(f"Block '{block}' not found. Available: {available}")

    mask = data[key].astype(bool)
    ch_names = [str(ch) for ch in data["ch_names"]]

    if exclude_bad:
        bad_set = {str(ch) for ch in data["bad_channels"]}
        good = np.array([ch not in bad_set for ch in ch_names])
        mask = mask[good]
        ch_names = [ch for ch, keep in zip(ch_names, good) if keep]

    return mask, ch_names

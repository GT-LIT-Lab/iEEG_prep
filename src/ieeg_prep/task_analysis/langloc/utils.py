"""Language localizer utilities: trial parsing and response vectors."""

from __future__ import annotations

import numpy as np

from ..localization import permutation_test  # noqa: F401  (re-exported for backward compat)

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
        Mean envelope across all word-window samples, per channel per trial.
    respvec_ideal : np.ndarray, shape (n_trials,)
        +1 for sentence trials, -1 for non-word trials.
    """
    from ..localization import compute_response_vector as _core

    sentence_code = event_codes["sentence"]
    nonword_code  = event_codes["non_word"]

    pos_trials: list[np.ndarray] = []
    neg_trials: list[np.ndarray] = []

    for t, trial in enumerate(trials):
        windows = []
        for start, end in trial["word_bounds"]:
            if end <= start:
                print(f"Warning: trial {t} has empty window [{start}, {end}), skipping")
                continue
            if end > envelope_data.shape[1]:
                print(f"Warning: trial {t} end_sample {end} exceeds envelope length, skipping")
                continue
            windows.append(envelope_data[:, start:end])
        if not windows:
            continue
        sig = np.concatenate(windows, axis=1)

        code = int(trial["word_codes"][0])
        if code == sentence_code:
            pos_trials.append(sig)
        elif code == nonword_code:
            neg_trials.append(sig)

    return _core(pos_trials, neg_trials)




def load_lang_mask(
    results_path,
    block: str = "superset",
    exclude_bad: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Load a language-responsive mask from a langloc results NPZ file.

    The mask is aligned to the envelope channel order used during the pipeline
    run, so it can be passed directly to
    :func:`~ieeg_prep.viz.langloc_plots.plot_sent_nw_timeseries` or used to index into
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

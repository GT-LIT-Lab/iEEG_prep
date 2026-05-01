"""Language localizer pipeline: load FIF + events, run permutation tests, update electrodes CSV."""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .utils import (
    DEFAULT_EVENT_CODES,
    get_trial_word_boundaries_from_block,
    compute_response_vector,
    permutation_test,
)
from ..utils import load_block


def _normalize_ch(name: str) -> str:
    m = re.match(r"([A-Za-z]+)-?0*([0-9]+)$", str(name))
    return f"{m.group(1)}{int(m.group(2))}" if m else str(name)


def _build_ch_index(seeg_names: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, ch in enumerate(seeg_names):
        index[ch] = i
        index[_normalize_ch(ch)] = i
    return index


def run_langloc_pipeline(
    envelope_fif_path,
    events_path,
    blocks_info_path,
    langloc_blocks: list[str],
    results_path,
    electrodes_csv_path=None,
    channel_col: str = "channel",
    event_codes: dict[str, int] | None = None,
    superset: bool = True,
    n_perm: int = 10000,
    seed: int = 42,
    threshold_pct: float = 95.0,
    n_words: int = 12,
    plot: bool = False,
    plot_output_dir=None,
    mni_coords_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
) -> dict:
    """Run the full language localizer pipeline.

    All channels in the envelope (including bad channels marked by the
    preprocessing pipeline) are included so that each electrode's language
    responsiveness is estimated independently.

    Parameters
    ----------
    envelope_fif_path : str or Path
        Preprocessed envelope FIF (output of run-prep).
    events_path : str or Path
        Preprocessed events array (.npy).
    blocks_info_path : str or Path
        Block metadata JSON (output of run-blocks).
    langloc_blocks : list[str]
        Labels of the langloc blocks to analyse (must be present in blocks_info).
    results_path : str or Path
        Destination for the NPZ results file.
    electrodes_csv_path : str or Path, optional
        Existing electrodes CSV to update with language mask columns.
    channel_col : str
        Column in ``electrodes_csv_path`` containing SEEG channel names.
    event_codes : dict[str, int], optional
        Name-to-code mapping. Defaults to :data:`DEFAULT_EVENT_CODES`.
    superset : bool
        If True, pool all blocks into a single combined permutation test and
        save results under the ``superset`` key.
    n_perm : int
        Number of label permutations for the null distribution.
    seed : int
        RNG seed for reproducibility.
    threshold_pct : float
        Null-distribution percentile used as the significance threshold.
    n_words : int
        Expected number of words per trial.
    plot : bool
        If True, save glass-brain plots of language vs. non-language electrodes
        (bad channels excluded) for each block and the superset.
        Requires ``electrodes_csv_path`` to be set.
    plot_output_dir : str or Path or None
        Directory for brain plot PNGs. Defaults to the parent of
        ``results_path`` when not specified.
    mni_coords_cols : tuple of str
        Column names in ``electrodes_csv_path`` for MNI x, y, z coordinates.

    Returns
    -------
    dict with keys:

    - ``ch_names`` list[str]: SEEG channel names (``_hg`` suffix stripped).
    - ``bad_channels`` list[str]: bad SEEG channel names.
    - ``per_block`` dict[str, dict]: per-block permutation test results.
    - ``superset`` dict or None: pooled test results, None if ``superset=False``
      or only one block was provided.
    """
    if event_codes is None:
        event_codes = DEFAULT_EVENT_CODES

    # Load envelope — ALL misc channels, bads included
    env_raw = mne.io.read_raw_fif(str(envelope_fif_path), preload=True, verbose=False)
    bad_ch_set = set(env_raw.info["bads"])

    all_picks = mne.pick_types(env_raw.info, misc=True, exclude=[])
    all_ch_names = [env_raw.ch_names[i] for i in all_picks]
    seeg_names = [ch.replace("_hg", "") for ch in all_ch_names]
    bad_seeg_names = [ch.replace("_hg", "") for ch in all_ch_names if ch in bad_ch_set]

    envelope_data = env_raw.get_data(picks=all_picks)  # (n_channels, n_samples)
    events = np.load(events_path)

    per_block: dict[str, dict] = {}
    respvecs: list[np.ndarray] = []
    respvec_ideals: list[np.ndarray] = []

    for label in langloc_blocks:
        print(f"Processing block: {label}")
        block = load_block(blocks_info_path, label, events)
        trials, bad_trials = get_trial_word_boundaries_from_block(
            block, event_codes=event_codes, n_words=n_words
        )
        if bad_trials:
            print(f"  Warning: {len(bad_trials)} malformed trial(s) skipped in '{label}'")

        respvec, respvec_ideal = compute_response_vector(trials, envelope_data, event_codes)
        true_corrs, null_corrs, p_values, is_lang, thresholds = permutation_test(
            respvec, respvec_ideal,
            n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )
        per_block[label] = {
            "true_corrs": true_corrs,
            "null_corrs": null_corrs,
            "p_values": p_values,
            "is_language_responsive": is_lang,
            "thresholds": thresholds,
        }
        respvecs.append(respvec)
        respvec_ideals.append(respvec_ideal)

    superset_result: dict | None = None
    if superset and len(langloc_blocks) > 1:
        print("Running superset permutation test...")
        combined_respvec = np.concatenate(respvecs, axis=1)
        combined_ideal = np.concatenate(respvec_ideals, axis=0)
        true_corrs, null_corrs, p_values, is_lang, thresholds = permutation_test(
            combined_respvec, combined_ideal,
            n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )
        superset_result = {
            "true_corrs": true_corrs,
            "null_corrs": null_corrs,
            "p_values": p_values,
            "is_language_responsive": is_lang,
            "thresholds": thresholds,
        }

    result = {
        "ch_names": seeg_names,
        "bad_channels": bad_seeg_names,
        "per_block": per_block,
        "superset": superset_result,
    }

    _save_npz(result, langloc_blocks, results_path)

    if electrodes_csv_path is not None:
        _update_electrodes_csv(result, langloc_blocks, electrodes_csv_path, channel_col)

    if plot:
        if electrodes_csv_path is None:
            print("Warning: plot=True but no electrodes_csv_path provided; skipping brain plots.")
        else:
            output_dir = Path(plot_output_dir) if plot_output_dir is not None else Path(results_path).parent
            _plot_langloc(result, langloc_blocks, electrodes_csv_path, output_dir, mni_coords_cols)

    return result


def _plot_langloc(
    result: dict,
    block_labels: list[str],
    electrodes_csv_path,
    plot_output_dir: Path,
    mni_coords_cols: tuple[str, str, str],
) -> None:
    from ...viz.csv2brainplot import run_from_dict

    mx_col, my_col, mz_col = mni_coords_cols

    plots = [
        {
            "title": f"LangLoc {lbl}",
            "filename": f"langloc_{lbl}_brain.png",
            "masked": True,
            "data": lbl,
            "colors": ["red", "steelblue"],
            "labels": ["Language", "Non-language"],
        }
        for lbl in block_labels
    ]
    if result["superset"] is not None:
        plots.append({
            "title": "LangLoc superset",
            "filename": "langloc_superset_brain.png",
            "masked": True,
            "data": "langloc_superset",
            "colors": ["red", "steelblue"],
            "labels": ["Language", "Non-language"],
        })

    run_from_dict({
        "csv": str(electrodes_csv_path),
        "coord_x_col": mx_col,
        "coord_y_col": my_col,
        "coord_z_col": mz_col,
        "exclude": "is_bad",
        "output_dir": str(plot_output_dir),
        "plots": plots,
    })


def _save_npz(result: dict, block_labels: list[str], results_path) -> None:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "ch_names": np.array(result["ch_names"]),
        "bad_channels": np.array(result["bad_channels"]),
        "block_labels": np.array(block_labels),
    }
    for label, res in result["per_block"].items():
        for key, val in res.items():
            arrays[f"{label}_{key}"] = val

    if result["superset"] is not None:
        for key, val in result["superset"].items():
            arrays[f"superset_{key}"] = val

    np.savez(results_path, **arrays)
    print(f"Saved results to {results_path}")


def _update_electrodes_csv(
    result: dict,
    block_labels: list[str],
    csv_path,
    channel_col: str,
) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    ch_index = _build_ch_index(result["ch_names"])

    def _col(values: np.ndarray) -> list[int]:
        out = []
        for raw_name in df[channel_col]:
            raw = str(raw_name)
            idx = ch_index.get(raw)
            if idx is None:
                idx = ch_index.get(_normalize_ch(raw))
            out.append(int(values[idx]) if idx is not None else 0)
        return out

    bad_set = set(result["bad_channels"])
    is_bad_arr = np.array([ch in bad_set for ch in result["ch_names"]])
    df["is_bad"] = _col(is_bad_arr)

    for label in block_labels:
        df[label] = _col(result["per_block"][label]["is_language_responsive"])

    if result["superset"] is not None:
        df["langloc_superset"] = _col(result["superset"]["is_language_responsive"])

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")

"""Semantic localizer pipeline for the MultiSem task."""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ...utils import load_coords
from ..localization import permutation_test
from ..utils import load_block
from .utils import (
    MULTISEM_EVENT_CODES,
    get_multisem_trials_from_block,
    compute_response_vector,
)
from .analysis import build_multisem_trial_tensor


def _normalize_ch(name: str) -> str:
    m = re.match(r"([A-Za-z]+)-?0*([0-9]+)$", str(name))
    return f"{m.group(1)}{int(m.group(2))}" if m else str(name)


def _build_ch_index(seeg_names: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, ch in enumerate(seeg_names):
        index[ch] = i
        index[_normalize_ch(ch)] = i
    return index


def run_multisem_pipeline(
    envelope_fif_path,
    events_path,
    blocks_info_path,
    multisem_blocks: list[str],
    results_path,
    electrodes_csv_path=None,
    channel_col: str = "channel",
    event_codes: dict[str, int] | None = None,
    superset: bool = True,
    n_perm: int = 10000,
    seed: int = 42,
    threshold_pct: float = 95.0,
    trials_per_condition: int = 6,
    plot: bool = False,
    plot_output_dir=None,
    mni_coords_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
) -> dict:
    """Run the full MultiSem semantic localizer pipeline.

    For each block, runs two Spearman permutation tests:
    semantic vs. perceptual for sentences, and semantic vs. perceptual for
    pictures.  The final per-block semantic mask is the intersection of the
    two.  When ``superset=True`` and multiple blocks are provided, trial
    response vectors are pooled across blocks and a combined test is run.

    Parameters
    ----------
    envelope_fif_path : str or Path
        Preprocessed envelope FIF (output of run-prep).
    events_path : str or Path
        Preprocessed events array (.npy).
    blocks_info_path : str or Path
        Block metadata JSON (output of run-blocks).
    multisem_blocks : list[str]
        Labels of the multisem blocks to analyse.
    results_path : str or Path
        Destination for the NPZ results file.
    electrodes_csv_path : str or Path, optional
        Existing electrodes CSV to update with semantic mask columns.
    channel_col : str
        Column in ``electrodes_csv_path`` containing SEEG channel names.
    event_codes : dict[str, int], optional
        Name-to-code mapping. Defaults to :data:`MULTISEM_EVENT_CODES`.
    superset : bool
        If True, pool all blocks into a combined permutation test.
    n_perm : int
        Number of label permutations for the null distribution.
    seed : int
        RNG seed for reproducibility.
    threshold_pct : float
        Null-distribution percentile used as the significance threshold.
    trials_per_condition : int
        Expected number of trials per condition per block (default 6).
    plot : bool
        If True, save glass-brain plots for each block and the superset.
        Requires ``electrodes_csv_path`` to be set.
    plot_output_dir : str or Path or None
        Directory for brain plot PNGs. Defaults to parent of ``results_path``.
    mni_coords_cols : tuple of str
        Column names in ``electrodes_csv_path`` for MNI x, y, z coordinates.

    Returns
    -------
    dict with keys:

    - ``ch_names`` list[str]: SEEG channel names.
    - ``bad_channels`` list[str]: bad SEEG channel names.
    - ``per_block`` dict[str, dict]: per-block results with ``sent``, ``pic``,
      and ``is_semantic_responsive`` sub-keys.
    - ``superset`` dict or None: pooled results, None if ``superset=False``
      or only one block was provided.
    """
    if event_codes is None:
        event_codes = MULTISEM_EVENT_CODES

    env_raw = mne.io.read_raw_fif(str(envelope_fif_path), preload=True, verbose=False)
    bad_ch_set = set(env_raw.info["bads"])

    all_picks = mne.pick_types(env_raw.info, misc=True, exclude=[])
    all_ch_names = [env_raw.ch_names[i] for i in all_picks]
    seeg_names = [ch.replace("_hg", "") for ch in all_ch_names]
    bad_seeg_names = [ch.replace("_hg", "") for ch in all_ch_names if ch in bad_ch_set]

    envelope_data = env_raw.get_data(picks=all_picks)
    events = np.load(events_path)

    per_block: dict[str, dict] = {}
    sent_rvs:    list[np.ndarray] = []
    sent_ideals: list[np.ndarray] = []
    pic_rvs:     list[np.ndarray] = []
    pic_ideals:  list[np.ndarray] = []

    for label in multisem_blocks:
        print(f"Processing block: {label}")
        block = load_block(blocks_info_path, label, events)
        trials, bad_trials = get_multisem_trials_from_block(
            block,
            event_codes=event_codes,
            trials_per_condition=trials_per_condition,
        )
        if bad_trials:
            print(f"  Warning: {len(bad_trials)} malformed trial(s) skipped in '{label}'")

        tensor, conditions = build_multisem_trial_tensor(trials, envelope_data)

        sent_rv, sent_ideal = compute_response_vector(
            tensor, conditions, "semantic_sent", "perceptual_sent"
        )
        pic_rv, pic_ideal = compute_response_vector(
            tensor, conditions, "semantic_pic", "perceptual_pic"
        )

        sent_corrs, sent_null, sent_p, sent_mask, sent_thresh = permutation_test(
            sent_rv, sent_ideal, n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )
        pic_corrs, pic_null, pic_p, pic_mask, pic_thresh = permutation_test(
            pic_rv, pic_ideal, n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )

        per_block[label] = {
            "sent": {
                "true_corrs":    sent_corrs,
                "null_corrs":    sent_null,
                "p_values":      sent_p,
                "is_responsive": sent_mask,
                "thresholds":    sent_thresh,
            },
            "pic": {
                "true_corrs":    pic_corrs,
                "null_corrs":    pic_null,
                "p_values":      pic_p,
                "is_responsive": pic_mask,
                "thresholds":    pic_thresh,
            },
            "is_semantic_responsive": sent_mask & pic_mask,
        }

        sent_rvs.append(sent_rv)
        sent_ideals.append(sent_ideal)
        pic_rvs.append(pic_rv)
        pic_ideals.append(pic_ideal)

    superset_result: dict | None = None
    if superset and len(multisem_blocks) > 1:
        print("Running superset permutation tests...")
        comb_sent_rv    = np.concatenate(sent_rvs,    axis=1)
        comb_sent_ideal = np.concatenate(sent_ideals, axis=0)
        comb_pic_rv     = np.concatenate(pic_rvs,     axis=1)
        comb_pic_ideal  = np.concatenate(pic_ideals,  axis=0)

        sent_corrs, sent_null, sent_p, sent_mask, sent_thresh = permutation_test(
            comb_sent_rv, comb_sent_ideal, n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )
        pic_corrs, pic_null, pic_p, pic_mask, pic_thresh = permutation_test(
            comb_pic_rv, comb_pic_ideal, n_perm=n_perm, threshold_pct=threshold_pct, seed=seed,
        )

        superset_result = {
            "sent": {
                "true_corrs":    sent_corrs,
                "null_corrs":    sent_null,
                "p_values":      sent_p,
                "is_responsive": sent_mask,
                "thresholds":    sent_thresh,
            },
            "pic": {
                "true_corrs":    pic_corrs,
                "null_corrs":    pic_null,
                "p_values":      pic_p,
                "is_responsive": pic_mask,
                "thresholds":    pic_thresh,
            },
            "is_semantic_responsive": sent_mask & pic_mask,
        }

    result = {
        "ch_names":     seeg_names,
        "bad_channels": bad_seeg_names,
        "per_block":    per_block,
        "superset":     superset_result,
    }

    _save_npz(result, multisem_blocks, results_path)

    if electrodes_csv_path is not None:
        _update_electrodes_csv(result, multisem_blocks, electrodes_csv_path, channel_col)

    if plot:
        if electrodes_csv_path is None:
            print("Warning: plot=True but no electrodes_csv_path provided; skipping brain plots.")
        else:
            output_dir = (
                Path(plot_output_dir) if plot_output_dir is not None
                else Path(results_path).parent
            )
            _plot_multisem(result, multisem_blocks, electrodes_csv_path, output_dir, mni_coords_cols, channel_col)

    return result


def _plot_multisem(
    result: dict,
    block_labels: list[str],
    electrodes_csv_path,
    plot_output_dir: Path,
    mni_coords_cols: tuple[str, str, str],
    channel_col: str,
) -> None:
    import pandas as pd
    from ...viz import plot_glass_brain

    mni = mni_coords_cols == ("mni_x", "mni_y", "mni_z")
    df = pd.read_csv(electrodes_csv_path)

    # Align CSV rows to result channel order, dropping bads and unknowns
    ch_index = _build_ch_index(result["ch_names"])
    bad_set  = set(result["bad_channels"])

    valid_rows: list[int] = []
    result_idx: list[int] = []
    for row_i, raw_name in enumerate(df[channel_col]):
        raw = str(raw_name)
        idx = ch_index.get(raw) if ch_index.get(raw) is not None else ch_index.get(_normalize_ch(raw))
        if idx is not None and result["ch_names"][idx] not in bad_set:
            valid_rows.append(row_i)
            result_idx.append(idx)

    if not valid_rows:
        print("  Warning: no matching channels found for brain plots, skipping.")
        return

    coords = load_coords(df, mni=mni)[valid_rows]
    ridx   = np.array(result_idx)

    plot_output_dir.mkdir(parents=True, exist_ok=True)

    def _four_groups(sent_m, pic_m, semantic_m):
        s = sent_m[ridx]
        p = pic_m[ridx]
        sem = semantic_m[ridx]
        return [sem, s & ~sem, p & ~sem, ~s & ~p]

    def _two_groups(semantic_m):
        sem = semantic_m[ridx]
        return [sem, ~sem]

    targets = [(lbl, result["per_block"][lbl], f"multisem_{lbl}") for lbl in block_labels]
    if result["superset"] is not None:
        targets.append(("superset", result["superset"], "multisem_superset"))

    for lbl, res, stem in targets:
        sent_m     = res["sent"]["is_responsive"]
        pic_m      = res["pic"]["is_responsive"]
        semantic_m = res["is_semantic_responsive"]

        # Plot 1: all contrasts with 4 groups
        plot_glass_brain(
            coords,
            _four_groups(sent_m, pic_m, semantic_m),
            masked=True,
            colors=["purple", "steelblue", "red", "lightgray"],
            labels=["Semantic (both)", "Sent only", "Pic only", "Non-responsive"],
            title=f"MultiSem {lbl} — all contrasts",
            output_path=str(plot_output_dir / f"{stem}_allcontrasts_brain.png"),
        )

        # Plot 2: semantic overlap only
        plot_glass_brain(
            coords,
            _two_groups(semantic_m),
            masked=True,
            colors=["purple", "lightgray"],
            labels=["Semantic", "Non-semantic"],
            title=f"MultiSem {lbl} — semantic overlap",
            output_path=str(plot_output_dir / f"{stem}_brain.png"),
        )

        print(f"  saved → {stem}_allcontrasts_brain.png")
        print(f"  saved → {stem}_brain.png")


def _save_npz(result: dict, block_labels: list[str], results_path) -> None:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "ch_names":     np.array(result["ch_names"]),
        "bad_channels": np.array(result["bad_channels"]),
        "block_labels": np.array(block_labels),
    }

    for label, res in result["per_block"].items():
        for contrast in ("sent", "pic"):
            for key, val in res[contrast].items():
                arrays[f"{label}_{contrast}_{key}"] = val
        arrays[f"{label}_is_semantic_responsive"] = res["is_semantic_responsive"]

    if result["superset"] is not None:
        for contrast in ("sent", "pic"):
            for key, val in result["superset"][contrast].items():
                arrays[f"superset_{contrast}_{key}"] = val
        arrays["superset_is_semantic_responsive"] = result["superset"]["is_semantic_responsive"]

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
            idx = ch_index.get(raw) if ch_index.get(raw) is not None else ch_index.get(_normalize_ch(raw))
            out.append(int(values[idx]) if idx is not None else 0)
        return out

    bad_set = set(result["bad_channels"])
    df["is_bad"] = _col(np.array([ch in bad_set for ch in result["ch_names"]]))

    for label in block_labels:
        df[f"{label}_sent"]     = _col(result["per_block"][label]["sent"]["is_responsive"])
        df[f"{label}_pic"]      = _col(result["per_block"][label]["pic"]["is_responsive"])
        df[f"{label}_semantic"] = _col(result["per_block"][label]["is_semantic_responsive"])

    if result["superset"] is not None:
        df["multisem_superset_sent"] = _col(result["superset"]["sent"]["is_responsive"])
        df["multisem_superset_pic"]  = _col(result["superset"]["pic"]["is_responsive"])
        df["multisem_superset"]      = _col(result["superset"]["is_semantic_responsive"])

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")

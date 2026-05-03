"""
MultiSem semantic localizer pipeline CLI.

Example config.json:
    {
        "envelope_path":       "/path/to/preprocessed_envelope.fif",
        "events_path":         "/path/to/preprocessed_events.npy",
        "blocks_info_path":    "/path/to/blocks_info.json",
        "multisem_blocks":     ["multisem1", "multisem2"],
        "results_path":        "/path/to/multisem_results.npz",
        "electrodes_csv_path": "/path/to/electrodes.csv",
        "channel_col":         "channel",
        "superset":            true,
        "n_perm":              10000,
        "seed":                42,
        "threshold_pct":       95.0,
        "trials_per_condition": 6,
        "plot":                false,
        "plot_output_dir":     "/path/to/figures",
        "mni_coords_cols":     ["mni_x", "mni_y", "mni_z"]
    }

Example usage:
    python -m ieeg_prep.task_analysis.multisem.run --config configs/EMOP0004/multisem_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from .pipeline import run_multisem_pipeline


def _print_job_report(result: dict, block_labels: list[str]) -> None:
    ch_names = result["ch_names"]
    bad_set  = set(result["bad_channels"])
    n_total  = len(ch_names)
    good_mask = np.array([ch not in bad_set for ch in ch_names])
    n_good = int(good_mask.sum())
    n_bad  = n_total - n_good

    print(f"\nChannels: {n_total} total  |  {n_good} good  |  {n_bad} bad\n")

    named_masks: dict[str, np.ndarray] = {}

    for label in block_labels:
        res = result["per_block"][label]
        for contrast in ("sent", "pic"):
            m = res[contrast]["is_responsive"]
            n   = int(m.sum())
            n_g = int((m & good_mask).sum())
            n_b = int((m & ~good_mask).sum())
            print(f"  {label} [{contrast}]:")
            print(f"    All channels : {n}/{n_total} semantic-responsive")
            print(f"    Good channels: {n_g}/{n_good}")
            print(f"    Bad channels : {n_b}/{n_bad}")

        sem = res["is_semantic_responsive"]
        n   = int(sem.sum())
        n_g = int((sem & good_mask).sum())
        n_b = int((sem & ~good_mask).sum())
        named_masks[label] = sem
        print(f"  {label} [semantic overlap]:")
        print(f"    All channels : {n}/{n_total} semantic-responsive")
        print(f"    Good channels: {n_g}/{n_good}")
        print(f"    Bad channels : {n_b}/{n_bad}")

    if result["superset"] is not None:
        sem = result["superset"]["is_semantic_responsive"]
        n   = int(sem.sum())
        n_g = int((sem & good_mask).sum())
        n_b = int((sem & ~good_mask).sum())
        named_masks["superset"] = sem
        label_str = "+".join(block_labels)
        print(f"  superset ({label_str}) [semantic overlap]:")
        print(f"    All channels : {n}/{n_total} semantic-responsive")
        print(f"    Good channels: {n_g}/{n_good}")
        print(f"    Bad channels : {n_b}/{n_bad}")

    keys = list(named_masks.keys())
    if len(keys) >= 2:
        print("\nPairwise overlaps (semantic masks):")
        for a, b in combinations(keys, 2):
            ma, mb = named_masks[a], named_masks[b]
            ov   = int((ma & mb).sum())
            ov_g = int((ma & mb & good_mask).sum())
            ov_b = int((ma & mb & ~good_mask).sum())
            print(f"  {a} ∩ {b}: {ov} total  |  {ov_g} good  |  {ov_b} bad")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MultiSem semantic localizer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
        help="Path to multisem config JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1

    with open(args.config) as f:
        cfg = json.load(f)

    required = ["envelope_path", "events_path", "blocks_info_path",
                "multisem_blocks", "results_path"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"Error: config missing required keys: {missing}", file=sys.stderr)
        return 1

    for key in ["envelope_path", "events_path", "blocks_info_path"]:
        if not Path(cfg[key]).exists():
            print(f"Error: {key} not found: {cfg[key]}", file=sys.stderr)
            return 1

    electrodes_csv = cfg.get("electrodes_csv_path")
    if electrodes_csv is not None and not Path(electrodes_csv).exists():
        print(f"Error: electrodes_csv_path not found: {electrodes_csv}", file=sys.stderr)
        return 1

    try:
        result = run_multisem_pipeline(
            envelope_fif_path    = cfg["envelope_path"],
            events_path          = cfg["events_path"],
            blocks_info_path     = cfg["blocks_info_path"],
            multisem_blocks      = cfg["multisem_blocks"],
            results_path         = cfg["results_path"],
            electrodes_csv_path  = electrodes_csv,
            channel_col          = cfg.get("channel_col", "channel"),
            superset             = bool(cfg.get("superset", True)),
            n_perm               = int(cfg.get("n_perm", 10000)),
            seed                 = int(cfg.get("seed", 42)),
            threshold_pct        = float(cfg.get("threshold_pct", 95.0)),
            trials_per_condition = int(cfg.get("trials_per_condition", 6)),
            plot                 = bool(cfg.get("plot", False)),
            plot_output_dir      = cfg.get("plot_output_dir"),
            mni_coords_cols      = tuple(cfg.get("mni_coords_cols", ("mni_x", "mni_y", "mni_z"))),
        )
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_job_report(result, cfg["multisem_blocks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

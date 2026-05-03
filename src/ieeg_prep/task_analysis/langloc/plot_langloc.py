"""Language localizer figure generation CLI.

Generates timeseries and mean-amplitude figures for one or more langloc
blocks, saving PNGs into per-section subdirectories.

The output_dir name should reflect the data type used (e.g.
``figures/SUBJECT/langloc_plots_zscore``).

Example config (JSON):
    {
        "envelope_path":        "/path/to/preprocessed_envelope.fif",
        "events_path":          "/path/to/preprocessed_events.npy",
        "blocks_info_path":     "/path/to/blocks_info.json",
        "langloc_results_path": "/path/to/langloc_results.npz",
        "langloc_blocks":       ["langloc1", "langloc2"],
        "frequency":            60,
        "output_dir":           "/path/to/figures/SUBJECT/langloc_plots_zscore",
        "plot_points":          true,
        "run_permutation_test": false,
        "plot_all_variants":    false,
        "n_permutations":       5000,
        "perm_seed":            42
    }

When plot_all_variants is true, all four combinations of
(plot_points: on/off) x (run_permutation_test: on/off) are generated
for the amplitude plots, ignoring the individual flags.

With a single block only self-plots are generated (no cross-validation,
no superset).  With two or more blocks, cross-validation pairs and a
pooled superset plot are also generated (superset requires that the
langloc pipeline was run with superset=True).

Output layout:
    output_dir/
        timeseries/
            {mask_block}_mask/
                {data_block}/
                    timeseries.png
        mean_amplitude/
            {mask_block}_mask/
                {data_block}/
                    mean_{variant}.png
                    diff_{variant}.png

    Self-plots always generated.  Cross-validation pairs and superset
    added when two or more blocks are provided (superset requires the
    langloc pipeline to have been run with superset=True).

Example usage:
    run-langloc-plots --config configs/EMOP0004/langloc_plots_config.json
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from ieeg_prep.task_analysis import (
    DEFAULT_EVENT_CODES,
    build_trial_tensor,
    get_trial_word_boundaries_from_block,
    load_block,
    load_lang_mask,
)
from ieeg_prep.viz import (
    plot_sent_nw_diff_amplitude,
    plot_sent_nw_mean_amplitude,
    plot_sent_nw_timeseries,
)

_REQUIRED_KEYS = [
    "envelope_path", "events_path", "blocks_info_path",
    "langloc_results_path", "langloc_blocks", "frequency", "output_dir",
]

_PATH_KEYS = [
    "envelope_path", "events_path", "blocks_info_path", "langloc_results_path",
]


def _variants(cfg: dict) -> list[tuple[bool, bool]]:
    if cfg.get("plot_all_variants", False):
        return [(pp, rp) for pp in (True, False) for rp in (True, False)]
    return [(cfg.get("plot_points", True), cfg.get("run_permutation_test", False))]


def _vsuffix(plot_points: bool, run_perm: bool) -> str:
    return f"{'points' if plot_points else 'nopoints'}_{'perm' if run_perm else 'noperm'}"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved → {path.name}")


def _collect_pairs(blocks: list[str], has_superset: bool) -> list[tuple[str, str]]:
    """(data_block, mask_block) pairs to plot.

    Always includes self-pairs.  Cross-validation pairs and the superset
    are added only when two or more blocks are present.
    """
    pairs = [(b, b) for b in blocks]
    if len(blocks) >= 2:
        for i, b1 in enumerate(blocks):
            for b2 in blocks[i + 1:]:
                pairs += [(b1, b2), (b2, b1)]
        if has_superset:
            pairs.append(("superset", "superset"))
    return pairs


def _run_timeseries(cfg: dict, data: dict, pairs: list[tuple[str, str]], out_dir: Path) -> None:
    freq = cfg["frequency"]
    for data_block, mask_block in pairs:
        sub = out_dir / f"{mask_block}_mask" / data_block
        sub.mkdir(parents=True, exist_ok=True)
        fname = "timeseries.png"
        plot_sent_nw_timeseries(
            data["tensors"][data_block],
            data["conds"][data_block],
            freq,
            word_onsets=data["wordstarts"][data_block],
            lang_mask=data["masks"][mask_block],
            title=f"{data_block} data | {mask_block} mask",
            output_path=str(sub / fname),
        )
        print(f"  saved → {mask_block}_mask/{data_block}/{fname}")


def _run_mean_amplitude(cfg: dict, data: dict, pairs: list[tuple[str, str]], out_dir: Path) -> None:
    variants = _variants(cfg)
    n_perm = cfg.get("n_permutations", 5000)
    seed = cfg.get("perm_seed", 42)

    for data_block, mask_block in pairs:
        tensor = data["tensors"][data_block]
        mask = data["masks"][mask_block]
        title = f"{data_block} data | {mask_block} mask"
        sub = out_dir / f"{mask_block}_mask" / data_block
        sub.mkdir(parents=True, exist_ok=True)

        for pp, rp in variants:
            suf = _vsuffix(pp, rp)
            kwargs = dict(
                title=title,
                group_labels=("Language", "Non-language"),
                plot_points=pp,
                run_permutation_test=rp,
                n_permutations=n_perm,
                perm_seed=seed,
            )
            _save(
                plot_sent_nw_mean_amplitude(tensor, mask, **kwargs),
                sub / f"mean_{suf}.png",
            )
            _save(
                plot_sent_nw_diff_amplitude(tensor, mask, **kwargs),
                sub / f"diff_{suf}.png",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate langloc timeseries and amplitude figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to langloc plots config JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1

    with open(args.config) as f:
        cfg = json.load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        print(f"Error: config missing required keys: {missing}", file=sys.stderr)
        return 1

    for key in _PATH_KEYS:
        if not Path(cfg[key]).exists():
            print(f"Error: {key} not found: {cfg[key]}", file=sys.stderr)
            return 1

    blocks = cfg["langloc_blocks"]

    print("Loading envelope data...")
    env = mne.io.read_raw_fif(cfg["envelope_path"], preload=True, verbose=False)
    env_data = env.copy().pick("misc", exclude="bads").get_data()

    events = np.load(cfg["events_path"])

    data: dict[str, dict] = {"tensors": {}, "conds": {}, "wordstarts": {}, "masks": {}}

    print(f"Loading blocks: {', '.join(blocks)}...")
    for b in blocks:
        block = load_block(cfg["blocks_info_path"], b, events)
        trial_info, _ = get_trial_word_boundaries_from_block(block)
        tensor, conds, ws = build_trial_tensor(trial_info, env_data, DEFAULT_EVENT_CODES)
        data["tensors"][b] = tensor
        data["conds"][b] = conds
        data["wordstarts"][b] = ws

    print("Loading language masks...")
    for b in blocks:
        mask, _ = load_lang_mask(cfg["langloc_results_path"], b)
        data["masks"][b] = mask

    has_superset = False
    if len(blocks) >= 2:
        print("Building superset data...")
        min_time = min(data["tensors"][b].shape[-1] for b in blocks)
        sup_tensor = np.concatenate(
            [data["tensors"][b][..., :min_time] for b in blocks], axis=1
        )
        ws_list = [data["wordstarts"][b] for b in blocks]
        sup_ws = np.mean(ws_list, axis=0) if all(
            len(ws) == len(ws_list[0]) for ws in ws_list
        ) else ws_list[0]
        data["tensors"]["superset"] = sup_tensor
        data["wordstarts"]["superset"] = sup_ws
        data["conds"]["superset"] = data["conds"][blocks[0]]

        try:
            sup_mask, _ = load_lang_mask(cfg["langloc_results_path"], "superset")
            data["masks"]["superset"] = sup_mask
            has_superset = True
        except Exception:
            print("  Warning: superset mask not found in results, skipping superset plots")

    pairs = _collect_pairs(blocks, has_superset)
    out_root = Path(cfg["output_dir"])

    print("\n--- Timeseries plots ---")
    _run_timeseries(cfg, data, pairs, out_root / "timeseries")

    print("\n--- Mean amplitude plots ---")
    _run_mean_amplitude(cfg, data, pairs, out_root / "mean_amplitude")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

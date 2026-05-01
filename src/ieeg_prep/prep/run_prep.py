"""
iEEG preprocessing pipeline CLI entry point.

Pipeline: load raw → channel setup → filter → CAR → HG envelope → normalize → save.

Example config.json:
    {
        "raw_path": "/path/to/file.edf",
        "output_dir": "/path/to/output",
        "preprocessing_config": {
            "sfreq_working": 400,
            "line_freq": 60,
            "ied_results_path": "/path/to/ied_results.npz",
            ...
        }
    }

Example usage:
    python -m ieeg_prep.prep.run_prep --config configs/prep_config.json
    python -m ieeg_prep.prep.run_prep --raw-path file.edf --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .pipeline import PreprocessingConfig, PreprocessingResult, run_preprocessing_pipeline


def _config_from_dict(d: dict) -> PreprocessingConfig:
    """Build PreprocessingConfig from a dict (e.g. parsed from JSON)."""
    c = d.get("preprocessing_config", d)
    return PreprocessingConfig(
        channel_renames=c.get("channel_renames", []),
        coordinates_csv=c.get("coordinates_csv", None),
        coord_channel_col=c.get("coord_channel_col", "label"),
        coord_x_col=c.get("coord_x_col", "mni_x"),
        coord_y_col=c.get("coord_y_col", "mni_y"),
        coord_z_col=c.get("coord_z_col", "mni_z"),
        stim_channel=c.get("stim_channel", "TRIG"),
        sfreq_working=float(c.get("sfreq_working", 400.0)),
        line_freq=float(c.get("line_freq", 60.0)),
        max_line_freq=float(c.get("max_line_freq", 180.0)),
        line_noise_peak_half_width=float(c.get("line_noise_peak_half_width", 2.0)),
        line_noise_bg_inner_gap=float(c.get("line_noise_bg_inner_gap", 3.0)),
        line_noise_bg_outer_width=float(c.get("line_noise_bg_outer_width", 10.0)),
        line_noise_threshold_sd=float(c.get("line_noise_threshold_sd", 5.0)),
        line_noise_n_fft=int(c.get("line_noise_n_fft", 4096)),
        ied_results_path=c.get("ied_results_path", None),
        filter_l_freq=float(c.get("filter_l_freq", 0.5)),
        filter_h_freq=float(c["filter_h_freq"]) if c.get("filter_h_freq") is not None else None,
        notch_freqs=[float(f) for f in c.get("notch_freqs", [60.0, 120.0, 180.0])],
        notch_method=c.get("notch_method", "iir"),
        hg_center_freqs=tuple(float(f) for f in c.get(
            "hg_center_freqs", [73.0, 79.5, 87.8, 96.9, 107.0, 118.1, 130.4, 144.0]
        )),
        hg_sigmas=tuple(float(s) for s in c.get(
            "hg_sigmas", [4.68, 4.92, 5.17, 5.43, 5.7, 5.99, 6.3, 6.62]
        )),
        hg_output=c.get("hg_output", "mean"),
        hg_ch_type_out=c.get("hg_ch_type_out", "misc"),
        outlier_percentile=float(c.get("outlier_percentile", 90.0)),
        outlier_multiplier=float(c.get("outlier_multiplier", 5.0)),
        hg_downsample_factor=int(c.get("hg_downsample_factor", 4)),
        normalize_zscore=bool(c.get("normalize_zscore", True)),
        normalize_minmax=bool(c.get("normalize_minmax", True)),
        target_sfreq=float(c.get("target_sfreq", 60.0)),
    )


def _config_to_dict(cfg: PreprocessingConfig) -> dict:
    """Serialize PreprocessingConfig to a JSON-serializable dict."""
    return {
        "channel_renames": cfg.channel_renames,
        "coordinates_csv": cfg.coordinates_csv,
        "coord_channel_col": cfg.coord_channel_col,
        "coord_x_col": cfg.coord_x_col,
        "coord_y_col": cfg.coord_y_col,
        "coord_z_col": cfg.coord_z_col,
        "stim_channel": cfg.stim_channel,
        "sfreq_working": cfg.sfreq_working,
        "line_freq": cfg.line_freq,
        "max_line_freq": cfg.max_line_freq,
        "line_noise_peak_half_width": cfg.line_noise_peak_half_width,
        "line_noise_bg_inner_gap": cfg.line_noise_bg_inner_gap,
        "line_noise_bg_outer_width": cfg.line_noise_bg_outer_width,
        "line_noise_threshold_sd": cfg.line_noise_threshold_sd,
        "line_noise_n_fft": cfg.line_noise_n_fft,
        "ied_results_path": cfg.ied_results_path,
        "filter_l_freq": cfg.filter_l_freq,
        "filter_h_freq": cfg.filter_h_freq,
        "notch_freqs": cfg.notch_freqs,
        "notch_method": cfg.notch_method,
        "hg_center_freqs": list(cfg.hg_center_freqs),
        "hg_sigmas": list(cfg.hg_sigmas),
        "hg_output": cfg.hg_output,
        "hg_ch_type_out": cfg.hg_ch_type_out,
        "outlier_percentile": cfg.outlier_percentile,
        "outlier_multiplier": cfg.outlier_multiplier,
        "hg_downsample_factor": cfg.hg_downsample_factor,
        "normalize_zscore": cfg.normalize_zscore,
        "normalize_minmax": cfg.normalize_minmax,
        "target_sfreq": cfg.target_sfreq,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iEEG preprocessing pipeline on an EDF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
        help="Path to config JSON. Loads raw_path, output_dir, and preprocessing_config.")
    parser.add_argument("--raw-path", type=Path, default=None,
        help="Path to raw EDF file. Required if not in --config.")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="Directory to save results. Default: current directory.")
    parser.add_argument("--ied-results-path", type=Path, default=None,
        help="Path to ied_results.npz. Overrides config value.")
    parser.add_argument("--return-intermediates", action="store_true",
        help="Save intermediate arrays (hg_raw, outlier_mask, etc.) in results.")

    ov = parser.add_argument_group("Config overrides")
    ov.add_argument("--sfreq-working", type=float, default=None)
    ov.add_argument("--line-freq", type=float, default=None)
    ov.add_argument("--line-noise-threshold-sd", type=float, default=None)
    ov.add_argument("--filter-l-freq", type=float, default=None)
    ov.add_argument("--notch-method", type=str, default=None,
        choices=["iir", "spectrum_fit"])
    ov.add_argument("--outlier-percentile", type=float, default=None)
    ov.add_argument("--outlier-multiplier", type=float, default=None)
    ov.add_argument("--hg-downsample-factor", type=int, default=None)
    ov.add_argument("--target-sfreq", type=float, default=None)
    ov.add_argument("--no-zscore", action="store_true",
        help="Disable z-score normalization.")
    ov.add_argument("--no-minmax", action="store_true",
        help="Disable min-max normalization.")

    return parser.parse_args()


def _resolve_config(
    args: argparse.Namespace,
) -> tuple[Path, Path, PreprocessingConfig, np.ndarray | None, bool]:
    """Resolve raw_path, output_dir, config, ied_mask from args + optional JSON."""
    config_dict: dict = {}
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config) as f:
            config_dict = json.load(f)
        if not isinstance(config_dict, dict):
            raise ValueError("Config JSON must be an object.")

    raw_path = args.raw_path or (
        Path(config_dict["raw_path"]) if "raw_path" in config_dict else None
    )
    if raw_path is None:
        raise ValueError(
            "raw_path is required. Provide via --raw-path or config JSON 'raw_path'."
        )
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    output_dir = args.output_dir or (
        Path(config_dict["output_dir"]) if "output_dir" in config_dict else Path.cwd()
    )
    output_dir = Path(output_dir)

    cfg = _config_from_dict(config_dict)

    # Apply scalar CLI overrides
    overrides = {
        "sfreq_working": args.sfreq_working,
        "line_freq": args.line_freq,
        "line_noise_threshold_sd": args.line_noise_threshold_sd,
        "filter_l_freq": args.filter_l_freq,
        "notch_method": args.notch_method,
        "outlier_percentile": args.outlier_percentile,
        "outlier_multiplier": args.outlier_multiplier,
        "hg_downsample_factor": args.hg_downsample_factor,
        "target_sfreq": args.target_sfreq,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.no_zscore:
        cfg.normalize_zscore = False
    if args.no_minmax:
        cfg.normalize_minmax = False

    # Resolve IED mask
    ied_results_path = args.ied_results_path or (
        Path(cfg.ied_results_path) if cfg.ied_results_path is not None else None
    )
    ied_mask: np.ndarray | None = None
    if ied_results_path is not None:
        if not Path(ied_results_path).exists():
            raise FileNotFoundError(f"IED results file not found: {ied_results_path}")
        ied_mask = np.load(ied_results_path)["channels_above_rate"]

    return raw_path, output_dir, cfg, ied_mask, args.return_intermediates


def _normalize_label(label: str) -> str:
    """Normalize electrode label to MNE channel name format (e.g. 'LAm-01' -> 'LAm1')."""
    m = re.match(r"([A-Za-z]+)-?0*([0-9]+)$", str(label))
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return label


def _apply_channel_renames(ch_names: list[str], renames: list[dict]) -> list[str]:
    """Apply electrode stem renames to a list of channel name strings.

    Mirrors the prefix-replace logic in :func:`~ieeg_prep.utils.rename_channels`
    so that IED result channel names can be aligned with renamed raw channels.
    """
    result = list(ch_names)
    for rule in renames:
        old, new = rule["name"], rule["new_name"]
        result = [
            ch.replace(old, new, 1) if ch.startswith(old) else ch
            for ch in result
        ]
    return result


def _save_electrode_table(
    result: PreprocessingResult,
    output_dir: Path,
    config: PreprocessingConfig,
) -> None:
    """Save a filtered electrode CSV with spike counts, spike rates, and bad labels.

    Only writes a file if ``config.coordinates_csv`` is set. Rows are filtered
    to electrodes present in the raw file. Columns added:

    - ``is_bad``: 1 if the channel is in bad_channels, 0 otherwise.
    - ``spike_count``: total IED spike count (requires ``config.ied_results_path``).
    - ``spike_rate``: IED spike rate in spikes/min (requires ``config.ied_results_path``).
    """
    if config.coordinates_csv is None:
        return

    df = pd.read_csv(config.coordinates_csv)
    df["_normalized"] = df[config.coord_channel_col].apply(_normalize_label)

    seeg_set = set(result.ch_names_seeg)
    df = df[df["_normalized"].isin(seeg_set)].copy()

    bad_set = set(result.bad_channels)
    df["is_bad"] = df["_normalized"].apply(lambda ch: int(ch in bad_set))

    if config.ied_results_path is not None:
        ied = np.load(config.ied_results_path, allow_pickle=True)
        ied_ch_names = [str(ch) for ch in ied["ch_names"]]
        if config.channel_renames:
            ied_ch_names = _apply_channel_renames(ied_ch_names, config.channel_renames)
        spike_counts = dict(zip(ied_ch_names, ied["channel_spike_counts"]))
        spike_rates = dict(zip(ied_ch_names, ied["channel_spike_rates"]))
        df["spike_count"] = df["_normalized"].map(spike_counts)
        df["spike_rate"] = df["_normalized"].map(spike_rates)

    df = df.drop(columns=["_normalized"])
    df.to_csv(output_dir / "electrodes.csv", index=False)


def save_results(
    result: PreprocessingResult,
    output_dir: Path,
    config: PreprocessingConfig,
) -> None:
    """Save the normalized envelope, events, electrode table, and metadata to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.env_norm_raw.save(output_dir / "preprocessed_envelope_ieeg.fif", overwrite=True)

    if result.events is not None:
        np.save(output_dir / "preprocessed_events.npy", result.events)

    metadata = {
        "sfreq_final": result.sfreq_final,
        "n_channels": len(result.ch_names_seeg),
        "ch_names_seeg": result.ch_names_seeg,
        "bad_channels": result.bad_channels,
        "n_bad_channels": len(result.bad_channels),
        "has_events": result.events is not None,
        "config": _config_to_dict(config),
    }
    with open(output_dir / "preprocessed_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _save_electrode_table(result, output_dir, config)


def main() -> int:
    args = _parse_args()
    try:
        raw_path, output_dir, config, ied_mask, return_intermediates = _resolve_config(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Loading: {raw_path}")
    raw = mne.io.read_raw_edf(raw_path, preload=True)

    print("Running preprocessing pipeline...")
    result = run_preprocessing_pipeline(
        raw,
        config=config,
        ied_mask=ied_mask,
        return_intermediates=return_intermediates,
    )

    print(f"Saving results to {output_dir}")
    save_results(result, output_dir, config)

    print(
        f"Done. sfreq={result.sfreq_final} Hz, "
        f"n_channels={len(result.ch_names_seeg)}, "
        f"n_bad={len(result.bad_channels)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

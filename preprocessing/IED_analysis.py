"""
IED (interictal epileptiform discharge) analysis.

CLI entry point: run from command line with a config.json or individual args.
Pipeline: load raw → label channels → run_interictal_pipeline → save results.

Example config.json:
    {
        "raw_path": "/path/to/file.edf",
        "output_dir": "/path/to/output",
        "interictal_config": {
            "sfreq_target": 200,
            "bandpass_low": 10,
            "bandpass_high": 60,
            "notch_method": "custom_biquad",
            ...
        }
    }

Example usage:
    python -m preprocessing.IED_analysis --config config.json
    python -m preprocessing.IED_analysis --raw-path file.edf --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

from .interictal import InterictalConfig, InterictalResult, run_interictal_pipeline

try:
    from ..utils import label_channels
except ImportError:
    # When run as __main__ (python -m preprocessing.IED_analysis), parent may not be a package
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_root))
    from utils import label_channels


def _config_from_dict(d: dict) -> InterictalConfig:
    """Build InterictalConfig from a dict (e.g. from JSON)."""
    ic = d.get("interictal_config", d)
    return InterictalConfig(
        sfreq_target=float(ic.get("sfreq_target", 200.0)),
        bandpass_low=float(ic.get("bandpass_low", 10.0)),
        bandpass_high=float(ic.get("bandpass_high", 60.0)),
        bandpass_order=int(ic.get("bandpass_order", 8)),
        bandpass_ftype=str(ic.get("bandpass_ftype", "cheby2")),
        bandpass_rs=float(ic.get("bandpass_rs", 60.0)),
        notch_freq=float(ic.get("notch_freq", 60.0)),
        notch_radius=float(ic.get("notch_radius", 0.985)),
        notch_method=str(ic.get("notch_method", "custom_biquad")),
        win_coeff=float(ic.get("win_coeff", 5.0)),
        overlap_coeff=float(ic.get("overlap_coeff", 4.0)),
        smooth_window_len=int(ic.get("smooth_window_len", 5)),
        threshold_k=float(ic.get("threshold_k", 3.65)),
        polyspike_union_sec=float(ic.get("polyspike_union_sec", 0.12)),
        trim_edges_sec=float(ic.get("trim_edges_sec", 1.0)),
    )


def _config_to_dict(cfg: InterictalConfig) -> dict:
    """Serialize InterictalConfig to a JSON-serializable dict."""
    return {
        "sfreq_target": cfg.sfreq_target,
        "bandpass_low": cfg.bandpass_low,
        "bandpass_high": cfg.bandpass_high,
        "bandpass_order": cfg.bandpass_order,
        "bandpass_ftype": cfg.bandpass_ftype,
        "bandpass_rs": cfg.bandpass_rs,
        "notch_freq": cfg.notch_freq,
        "notch_radius": cfg.notch_radius,
        "notch_method": cfg.notch_method,
        "win_coeff": cfg.win_coeff,
        "overlap_coeff": cfg.overlap_coeff,
        "smooth_window_len": cfg.smooth_window_len,
        "threshold_k": cfg.threshold_k,
        "polyspike_union_sec": cfg.polyspike_union_sec,
        "trim_edges_sec": cfg.trim_edges_sec,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IED (interictal epileptiform discharge) analysis on an EDF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json. If given, loads raw_path, output_dir, and interictal_config from it.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to raw EDF file. Required if --config is not given or config lacks raw_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results. Default: current directory.",
    )
    # InterictalConfig args (override config.json when provided)
    ic_group = parser.add_argument_group("Interictal pipeline config (override config.json)")
    ic_group.add_argument("--sfreq-target", type=float, default=None)
    ic_group.add_argument("--bandpass-low", type=float, default=None)
    ic_group.add_argument("--bandpass-high", type=float, default=None)
    ic_group.add_argument("--bandpass-order", type=int, default=None)
    ic_group.add_argument("--bandpass-ftype", type=str, default=None)
    ic_group.add_argument("--bandpass-rs", type=float, default=None)
    ic_group.add_argument("--notch-freq", type=float, default=None)
    ic_group.add_argument("--notch-radius", type=float, default=None)
    ic_group.add_argument("--notch-method", type=str, choices=["custom_biquad", "mne"], default=None)
    ic_group.add_argument("--win-coeff", type=float, default=None)
    ic_group.add_argument("--overlap-coeff", type=float, default=None)
    ic_group.add_argument("--smooth-window-len", type=int, default=None)
    ic_group.add_argument("--threshold-k", type=float, default=None)
    ic_group.add_argument("--polyspike-union-sec", type=float, default=None)
    ic_group.add_argument("--trim-edges-sec", type=float, default=None)
    parser.add_argument(
        "--return-intermediates",
        action="store_true",
        help="Save intermediate arrays (threshold, mu_interp, etc.) in results.",
    )
    parser.add_argument(
        "--channels-above-rate-thresh",
        type=float,
        default=None,
        help="Rate threshold (spikes/min) for channels_above_rate boolean. Saved in results. Default: 6.5.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> tuple[Path, Path, InterictalConfig, bool, float]:
    """
    Resolve raw_path, output_dir, InterictalConfig from args and optional config file.
    Raises if raw_path cannot be determined.
    """
    config_dict: dict = {}
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config) as f:
            config_dict = json.load(f)
        if not isinstance(config_dict, dict):
            raise ValueError("Config JSON must be an object.")

    raw_path = args.raw_path
    if raw_path is None:
        raw_path = config_dict.get("raw_path")
        if raw_path is not None:
            raw_path = Path(raw_path)
    if raw_path is None or (isinstance(raw_path, Path) and str(raw_path) == ""):
        raise ValueError(
            "raw_path is required. Provide it via --raw-path or in config.json under 'raw_path'."
        )
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    output_dir = args.output_dir
    if output_dir is None:
        od = config_dict.get("output_dir")
        output_dir = Path(od) if od is not None else Path.cwd()
    else:
        output_dir = Path(output_dir)

    cfg = _config_from_dict(config_dict)
    # Override with CLI args
    overrides = {
        "sfreq_target": args.sfreq_target,
        "bandpass_low": args.bandpass_low,
        "bandpass_high": args.bandpass_high,
        "bandpass_order": args.bandpass_order,
        "bandpass_ftype": args.bandpass_ftype,
        "bandpass_rs": args.bandpass_rs,
        "notch_freq": args.notch_freq,
        "notch_radius": args.notch_radius,
        "notch_method": args.notch_method,
        "win_coeff": args.win_coeff,
        "overlap_coeff": args.overlap_coeff,
        "smooth_window_len": args.smooth_window_len,
        "threshold_k": args.threshold_k,
        "polyspike_union_sec": args.polyspike_union_sec,
        "trim_edges_sec": args.trim_edges_sec,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    rate_thresh = args.channels_above_rate_thresh
    if rate_thresh is None:
        rate_thresh = float(config_dict.get("channels_above_rate_thresh", 6.5))
    return raw_path, output_dir, cfg, args.return_intermediates, rate_thresh


def save_results(
    result: InterictalResult,
    output_dir: Path,
    raw: mne.io.Raw,
    config: InterictalConfig,
    channels_above_rate_thresh: float = 6.5,
) -> None:
    """Save spike_markers, channel counts/rates, channels_above_rate, and metadata to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeg_picks = mne.pick_types(raw.info, seeg=True)
    ch_names = [raw.ch_names[i] for i in seeg_picks]

    channels_above_rate = result.channels_above_rate(channels_above_rate_thresh)

    np.savez_compressed(
        output_dir / "ied_results.npz",
        spike_markers=result.spike_markers,
        channel_spike_counts=result.channel_spike_counts,
        channel_spike_rates=result.channel_spike_rates,
        channels_above_rate=channels_above_rate,
        ch_names=np.array(ch_names, dtype=object),
        sfreq=result.sfreq,
    )

    metadata = {
        "sfreq": result.sfreq,
        "time_min": result.time_min,
        "n_spikes_total": result.n_spikes_total,
        "n_channels": result.spike_markers.shape[0],
        "n_samples": result.spike_markers.shape[1],
        "ch_names": ch_names,
        "channels_above_rate_thresh": channels_above_rate_thresh,
        "n_channels_above_rate": int(channels_above_rate.sum()),
        "config": _config_to_dict(config),
    }
    if result.threshold is not None:
        np.save(output_dir / "threshold.npy", result.threshold)
    if result.mu_interp is not None:
        np.save(output_dir / "mu_interp.npy", result.mu_interp)
    if result.sigma_interp is not None:
        np.save(output_dir / "sigma_interp.npy", result.sigma_interp)

    with open(output_dir / "ied_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    args = _parse_args()
    try:
        raw_path, output_dir, config, return_intermediates, rate_thresh = _resolve_config(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Loading: {raw_path}")
    raw = mne.io.read_raw_edf(raw_path, preload=True)
    raw = label_channels(raw)

    print("Running interictal pipeline...")
    result = run_interictal_pipeline(raw, config=config, return_intermediates=return_intermediates)

    print(f"Saving results to {output_dir}")
    save_results(result, output_dir, raw, config, channels_above_rate_thresh=rate_thresh)

    print(f"Done. n_spikes_total={result.n_spikes_total}, time_min={result.time_min:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

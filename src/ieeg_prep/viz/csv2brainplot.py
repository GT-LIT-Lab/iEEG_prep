"""
csv2brainplot — batch glass-brain plotter driven by a JSON config.

Usage
-----
    csv2brainplot path/to/config.json

Config schema
-------------
{
    "csv": "<path to electrode CSV>",

    // coordinate columns (defaults shown)
    "coord_x_col": "mni_x",
    "coord_y_col": "mni_y",
    "coord_z_col": "mni_z",

    // optional: column name or list of column names; rows where any == 1 are
    // excluded from all plots (e.g. bad electrodes).
    "exclude": "bad_channel",          // or ["bad_channel", "out_of_brain"]

    // optional: directory where figures are saved; omit to display interactively
    "output_dir": "path/to/output/",

    "plots": [
        // --- mask mode ---
        {
            "title": "Language Electrodes",
            "filename": "lang_brain.png",   // optional; derived from title if omitted
            "masked": true,
            "data": "lang_col",             // binary (0/1) column in the CSV
            "colors": ["red", "blue"],      // colors in True (1), False (0) order
            "labels": ["Lang+", "Control"],
            "s": 20
        },
        // --- continuous mode ---
        {
            "title": "Gamma Power",
            "filename": "gamma_brain.png",
            "masked": false,
            "data": "gamma_col",            // scalar column in the CSV
            "cmap": "hot",
            "vmin": null,
            "vmax": null,
            "center": null,
            "symmetric": false,
            "colorbar": true,
            "s": 20
        }
    ]
}

Notes
-----
* If ``output_dir`` is omitted, each figure is shown interactively.
* ``filename`` inside a plot entry is optional; when absent the title is
  sanitised and used as the filename (spaces → underscores, ``.png`` appended).
* All ``plot_glass_brain`` keyword arguments are supported directly in each
  plot entry except ``data`` (replaced by column name(s)) and ``output_path``
  (derived from ``output_dir`` + ``filename``).
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from ieeg_prep.viz import plot_glass_brain

# Keys that are consumed by the script and must not be forwarded to
# plot_glass_brain as kwargs.
_SCRIPT_KEYS = {"data", "filename", "masked"}

# Recognised plot_glass_brain keyword arguments (excluding coords/data/masked).
_PLOT_KWARGS = {
    "colors", "labels", "cmap", "vmin", "vmax", "center",
    "symmetric", "s", "title", "colorbar",
}


def _sanitise_filename(title: str) -> str:
    """Convert a plot title into a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", title).strip()
    safe = re.sub(r"[\s]+", "_", safe)
    return safe + ".png"


def run_from_dict(cfg: dict) -> None:
    """Run the batch plotter from a config dict (same schema as the JSON file)."""
    csv_path = cfg["csv"]
    x_col = cfg.get("coord_x_col", "mni_x")
    y_col = cfg.get("coord_y_col", "mni_y")
    z_col = cfg.get("coord_z_col", "mni_z")
    output_dir = cfg.get("output_dir")

    df = pd.read_csv(csv_path)

    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in {csv_path}")

    exclude_spec = cfg.get("exclude")
    if exclude_spec is not None:
        if isinstance(exclude_spec, str):
            exclude_spec = [exclude_spec]
        for col in exclude_spec:
            if col not in df.columns:
                raise ValueError(f"Exclude column '{col}' not found in {csv_path}")
        exclude_mask = np.zeros(len(df), dtype=bool)
        for col in exclude_spec:
            exclude_mask |= df[col].to_numpy() == 1
        df = df[~exclude_mask].reset_index(drop=True)

    coords = df[[x_col, y_col, z_col]].to_numpy(dtype=float)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    plots = cfg.get("plots", [])
    if not plots:
        print("No plots defined in config — nothing to do.", file=sys.stderr)
        return

    for i, plot_cfg in enumerate(plots):
        masked = plot_cfg.get("masked", False)
        data_spec = plot_cfg["data"]

        # --- resolve data from CSV columns ---
        if masked:
            if not isinstance(data_spec, str):
                raise ValueError(
                    f"Plot {i}: 'data' must be a single column name string when masked=true"
                )
            if data_spec not in df.columns:
                raise ValueError(
                    f"Plot {i}: mask column '{data_spec}' not found in {csv_path}"
                )
            col_vals = df[data_spec].to_numpy()
            unique = set(col_vals)
            if not unique.issubset({0, 1}):
                raise ValueError(
                    f"Plot {i}: column '{data_spec}' is not a binary mask — "
                    f"found values: {sorted(unique)}"
                )
            # colors are in True (1), False (0) order
            data = [col_vals == 1, col_vals == 0]
        else:
            if not isinstance(data_spec, str):
                raise ValueError(
                    f"Plot {i}: 'data' must be a column name string when masked=false"
                )
            if data_spec not in df.columns:
                raise ValueError(
                    f"Plot {i}: data column '{data_spec}' not found in {csv_path}"
                )
            data = df[data_spec].to_numpy(dtype=float)

        # --- build output path ---
        output_path = None
        if output_dir is not None:
            filename = plot_cfg.get("filename") or _sanitise_filename(
                plot_cfg.get("title", f"plot_{i}")
            )
            output_path = os.path.join(output_dir, filename)

        # --- collect remaining kwargs for plot_glass_brain ---
        kwargs = {k: v for k, v in plot_cfg.items() if k in _PLOT_KWARGS}

        print(f"  Plotting: {plot_cfg.get('title', f'plot_{i}')}", flush=True)
        plot_glass_brain(
            coords,
            data,
            masked=masked,
            output_path=output_path,
            **kwargs,
        )

        if output_path:
            print(f"    Saved → {output_path}", flush=True)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    run_from_dict(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="csv2brainplot",
        description="Batch glass-brain plotter driven by a JSON config.",
    )
    parser.add_argument("config", help="Path to the JSON config file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

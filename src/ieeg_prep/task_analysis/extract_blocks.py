"""
Block extraction and labeling CLI entry point.

Loads a preprocessed events array, extracts start/end blocks, and saves
labeled block metadata to JSON.

Example config.json:
    {
        "events_path": "/path/to/preprocessed_events.npy",
        "output_path": "/path/to/block_info.json",
        "labels": ["rest", "language", "motor"],
        "start_code": 1,
        "end_code": 2,
        "ignore_incomplete": true,
        "confirm_scan_plan_consulted": true
    }

Example usage:
    python -m ieeg_prep.task_analysis.extract_blocks --config configs/EMOP0636/blocks_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .utils import extract_blocks, save_block_info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract blocks from preprocessed events and save labeled metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
        help="Path to block config JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    with open(args.config) as f:
        cfg = json.load(f)

    events_path = Path(cfg["events_path"])
    if not events_path.exists():
        print(f"Error: events file not found: {events_path}", file=sys.stderr)
        return 1

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = cfg["labels"]
    start_code = int(cfg.get("start_code", 1))
    end_code = int(cfg.get("end_code", 2))
    ignore_incomplete = bool(cfg.get("ignore_incomplete", True))
    confirm_scan_plan_consulted = bool(cfg.get("confirm_scan_plan_consulted", False))

    print(f"Loading events: {events_path}")
    events = np.load(events_path)

    print("Extracting blocks...")
    blocks = extract_blocks(events, start_code=start_code, end_code=end_code)

    n_complete = sum(b["complete"] for b in blocks)
    print(f"Found {len(blocks)} block(s) total, {n_complete} complete.")

    try:
        save_block_info(
            blocks,
            labels=labels,
            output_path=output_path,
            ignore_incomplete=ignore_incomplete,
            confirm_scan_plan_consulted=confirm_scan_plan_consulted,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("\n--- blocks_info.json ---")
    print(output_path.read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())

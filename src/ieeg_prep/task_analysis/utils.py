import json
from pathlib import Path

import numpy as np


def extract_blocks(events, start_code=1, end_code=2):
    """
    Extract blocks of events between start_code and end_code.

    Parameters
    ----------
    events : np.ndarray
        MNE event array of shape (n_events, 3), where each row is
        [sample, prev_id, event_id].
    start_code : int
        Event code marking the start of a block.
    end_code : int
        Event code marking the end of a block.

    Returns
    -------
    blocks : list of dict
        Each dict contains:

        - ``start_event_idx`` (int): index into ``events`` of the start trigger.
        - ``end_event_idx`` (int or None): index of the end trigger, or None if
          the block was never closed.
        - ``start_sample`` (int): sample number of the start trigger.
        - ``end_sample`` (int or None): sample number of the end trigger.
        - ``events`` (np.ndarray): slice of the event array for this block,
          shape (n_block_events, 3).
        - ``complete`` (bool): True when both a start and end trigger were found.

    Notes
    -----
    Warns (via ``print``) when a start trigger appears before the previous
    block is closed, or when an end trigger appears without a matching start.
    In the former case the unclosed block is saved with ``complete=False``
    before opening a new one.
    """
    blocks = []
    open_start_idx = None

    for i, event in enumerate(events):
        sample, _, code = event

        if code == start_code:
            if open_start_idx is not None:
                print(f"Warning: found start at event {i} before closing previous block")
                blocks.append({
                    "start_event_idx": open_start_idx,
                    "end_event_idx": None,
                    "start_sample": events[open_start_idx, 0],
                    "end_sample": None,
                    "events": events[open_start_idx:i].copy(),
                    "complete": False,
                })
            open_start_idx = i

        elif code == end_code:
            if open_start_idx is not None:
                blocks.append({
                    "start_event_idx": open_start_idx,
                    "end_event_idx": i,
                    "start_sample": events[open_start_idx, 0],
                    "end_sample": sample,
                    "events": events[open_start_idx : i + 1].copy(),
                    "complete": True,
                })
                open_start_idx = None
            else:
                print(f"Warning: found end at event {i} without a matching start")

    if open_start_idx is not None:
        blocks.append({
            "start_event_idx": open_start_idx,
            "end_event_idx": None,
            "start_sample": events[open_start_idx, 0],
            "end_sample": None,
            "events": events[open_start_idx:].copy(),
            "complete": False,
        })

    return blocks


def save_block_info(blocks, labels, output_path, ignore_incomplete=True, confirm_scan_plan_consulted=False):
    """
    Save block metadata and task labels to a JSON file.

    Parameters
    ----------
    blocks : list of dict
        Output of :func:`extract_blocks`.
    labels : list of str
        Task label for each block. Must match the number of blocks being saved:
        when ``ignore_incomplete=True`` this is the number of complete blocks;
        when ``ignore_incomplete=False`` this is the total number of blocks.
    output_path : str or Path
        Destination path for the JSON file.
    ignore_incomplete : bool
        If True (default), incomplete blocks are excluded from the output.
        The length of ``labels`` must match the number of complete blocks.
        If False, all blocks are saved and ``labels`` must match the total
        number of blocks including incomplete ones.
    confirm_scan_plan_consulted : bool
        Must be set to True to proceed. This forces the caller to acknowledge
        that labels were derived from the scan plan rather than guessed.

    Raises
    ------
    ValueError
        If ``confirm_scan_plan_consulted`` is False.
    ValueError
        If the number of labels does not match the number of blocks being saved.
    """
    if not confirm_scan_plan_consulted:
        raise ValueError(
            "You must consult the scan plan before providing block labels. "
            "Once you have done so, set confirm_scan_plan_consulted=True to proceed."
        )

    target_blocks = [b for b in blocks if b["complete"]] if ignore_incomplete else blocks

    if len(labels) != len(target_blocks):
        context = "complete " if ignore_incomplete else ""
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match the number of "
            f"{context}blocks ({len(target_blocks)})."
        )

    records = []
    for block, label in zip(target_blocks, labels):
        records.append({
            "label": label,
            "complete": block["complete"],
            "start_sample": int(block["start_sample"]),
            "end_sample": int(block["end_sample"]) if block["end_sample"] is not None else None,
            "start_event_idx": int(block["start_event_idx"]),
            "end_event_idx": int(block["end_event_idx"]) if block["end_event_idx"] is not None else None,
            "n_events": len(block["events"]),
        })

    Path(output_path).write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} block(s) to {output_path}")


def load_block(
    blocks_path,
    label: str,
    events: np.ndarray,
) -> dict:
    """Load a saved block by label and reconstruct its events array.

    Parameters
    ----------
    blocks_path : str or Path
        Path to the JSON file written by :func:`save_block_info`.
    label : str
        Block label to load (e.g. ``"langloc1"``).
    events : np.ndarray, shape (n_events, 3)
        Full preprocessed MNE events array used to reconstruct the block slice.

    Returns
    -------
    block : dict
        Dictionary compatible with
        :func:`~ieeg_prep.task_analysis.langloc.get_trial_word_boundaries_from_block`,
        containing:

        - ``label`` (str)
        - ``complete`` (bool)
        - ``start_sample`` (int)
        - ``end_sample`` (int or None)
        - ``start_event_idx`` (int)
        - ``end_event_idx`` (int or None)
        - ``events`` (np.ndarray): slice of the full events array for this block.

    Raises
    ------
    KeyError
        If no block with the given label exists in the JSON.
    """
    records = json.loads(Path(blocks_path).read_text())

    matches = [r for r in records if r["label"] == label]
    if not matches:
        available = [r["label"] for r in records]
        raise KeyError(
            f"Label '{label}' not found. Available labels: {available}"
        )

    record = matches[0]
    start_idx = record["start_event_idx"]
    end_idx = record["end_event_idx"]

    block_events = (
        events[start_idx : end_idx + 1].copy()
        if end_idx is not None
        else events[start_idx:].copy()
    )

    return {
        "label": record["label"],
        "complete": record["complete"],
        "start_sample": record["start_sample"],
        "end_sample": record["end_sample"],
        "start_event_idx": start_idx,
        "end_event_idx": end_idx,
        "events": block_events,
    }

"""MultiSem task utilities: trial parsing."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MULTISEM_EVENT_CODES: dict[str, int] = {
    "experiment_start": 1,
    "experiment_end":   2,
    "semantic_pic":     3,
    "semantic_sent":    4,
    "perceptual_pic":   5,
    "perceptual_sent":  6,
    "notask_pic":       7,
    "notask_sent":      8,
    "trial":            9,
    "fixation":         10,
    "subject_resp":     11,
    "ITI":              12,
}

# The six condition triggers for the standard MultiSem paradigm.
# Pass a custom list to get_multisem_trials_from_block when using a
# different set of conditions.
MULTISEM_CONDITION_KEYS: list[str] = [
    "semantic_pic",
    "semantic_sent",
    "perceptual_pic",
    "perceptual_sent",
    "notask_pic",
    "notask_sent",
]


def get_multisem_trials_from_block(
    block: dict,
    event_codes: dict[str, int] | None = None,
    condition_keys: list[str] | None = None,
    trial_key: str = "trial",
    iti_key: str = "ITI",
    skip_keys: list[str] | None = None,
    trials_per_condition: int = 6,
) -> tuple[list[dict], list[dict]]:
    """Parse one block dictionary into per-trial records for the MultiSem task.

    The MultiSem paradigm presents stimuli in condition groups: a condition
    trigger precedes a sequence of ``trial`` / ``ITI`` pairs.  Each trial
    epoch spans from the trial trigger up to (but not including) the next
    trial or condition trigger, so the epoch includes the ITI period.

    Fixation triggers that appear between condition groups within the same
    block are silently stepped over when searching for the trial boundary.
    Any code that is not a condition trigger, trial trigger, or ITI trigger
    is similarly stepped over — including response triggers.

    After parsing, each condition group is validated against
    ``trials_per_condition``.  Groups with the wrong trial count are moved
    wholesale to ``bad_trials``, mirroring the ``n_words`` check in the
    language localizer parser.

    Parameters
    ----------
    block : dict
        Block dictionary as returned by
        :func:`~ieeg_prep.task_analysis.extract_blocks`.
    event_codes : dict[str, int], optional
        Name-to-code mapping.  Defaults to :data:`MULTISEM_EVENT_CODES`.
    condition_keys : list of str, optional
        Keys in *event_codes* that identify condition-block triggers.
        Defaults to :data:`MULTISEM_CONDITION_KEYS` (the six standard
        MultiSem conditions, codes 3–8).  Pass a custom list when using
        a different set of conditions.
    trial_key : str
        Key in *event_codes* for the per-trial trigger (default ``"trial"``).
    iti_key : str
        Key in *event_codes* for the ITI trigger (default ``"ITI"``).
    skip_keys : list of str, optional
        Keys in *event_codes* to silently ignore during parsing — typically
        response triggers (e.g. ``["subject_resp"]``).  Defaults to an empty
        list.
    trials_per_condition : int
        Expected number of trials per condition group (default 6).  Condition
        groups with a different count are moved to ``bad_trials``.

    Returns
    -------
    trials : list of dict
        One entry per valid trial, in temporal order.  Each dict contains:

        - ``trial_number`` (int): sequential index across all valid trials
        - ``condition_code`` (int): event code of the active condition trigger
        - ``condition_label`` (str): human-readable condition name
        - ``condition_rep`` (int): 0-indexed repetition count for this condition
        - ``trial_within_condition`` (int): 0-indexed position within the condition group
        - ``trial_start_sample`` (int): sample of the trial trigger
        - ``iti_start_sample`` (int): sample of the ITI trigger
        - ``trial_end_sample`` (int): sample of the next trial or condition trigger
          (exclusive epoch boundary); equals ``block["end_sample"]`` for the last trial
        - ``trial_start_event_idx_local`` (int): index within the block events array
        - ``trial_start_event_idx_global`` (int): index in the full events array
        - ``iti_event_idx_local`` (int)
        - ``iti_event_idx_global`` (int)

    bad_trials : list of dict
        Trials that could not be parsed cleanly, with a ``"reason"`` field:

        - ``"no preceding condition trigger"``
        - ``"missing ITI trigger"``
        - ``"wrong trial count in condition group"`` (includes ``found_n_trials``
          and ``expected_n_trials``)
    """
    if event_codes is None:
        event_codes = MULTISEM_EVENT_CODES
    if condition_keys is None:
        condition_keys = MULTISEM_CONDITION_KEYS
    if skip_keys is None:
        skip_keys = []

    trial_code      = event_codes[trial_key]
    iti_code        = event_codes[iti_key]
    skip_set        = {event_codes[k] for k in skip_keys}
    condition_codes = {event_codes[k] for k in condition_keys}
    code_to_label   = {event_codes[k]: k for k in condition_keys}
    boundary_codes  = condition_codes | {trial_code}

    events                = block["events"]
    block_start_event_idx = block["start_event_idx"]
    block_end_sample      = block["end_sample"]
    n_events              = len(events)

    candidate_trials: list[dict] = []
    bad_trials:       list[dict] = []

    current_condition_code:  int | None = None
    current_condition_label: str | None = None
    condition_rep:           dict[int, int] = {}  # code → 0-indexed rep count
    trial_within_condition:  int = 0

    i = 0
    while i < n_events:
        code = events[i, 2]

        if code in skip_set:
            i += 1
            continue

        if code in condition_codes:
            current_condition_code  = code
            current_condition_label = code_to_label[code]
            condition_rep[code]     = condition_rep.get(code, -1) + 1
            trial_within_condition  = 0
            i += 1
            continue

        if code != trial_code:
            # Fixation, experiment start/end, or other non-boundary events.
            i += 1
            continue

        # --- Trial trigger found ---
        trial_start_idx_local  = i
        trial_start_idx_global = block_start_event_idx + i
        trial_start_sample     = int(events[i, 0])

        # Scan forward: collect ITI position and locate next boundary.
        iti_idx_local:    int | None = None
        iti_idx_global:   int | None = None
        iti_sample:       int | None = None
        trial_end_sample: int        = block_end_sample

        j = i + 1
        while j < n_events:
            next_code = events[j, 2]

            if next_code in skip_set:
                j += 1
                continue

            if next_code == iti_code and iti_idx_local is None:
                iti_idx_local  = j
                iti_idx_global = block_start_event_idx + j
                iti_sample     = int(events[j, 0])
                j += 1
                continue

            if next_code in boundary_codes:
                trial_end_sample = int(events[j, 0])
                break

            j += 1

        if current_condition_code is None:
            bad_trials.append({
                "block_start_event_idx":        int(block_start_event_idx),
                "trial_start_event_idx_local":  int(trial_start_idx_local),
                "trial_start_event_idx_global": int(trial_start_idx_global),
                "reason": "no preceding condition trigger",
            })
        elif iti_idx_local is None:
            bad_trials.append({
                "block_start_event_idx":        int(block_start_event_idx),
                "trial_start_event_idx_local":  int(trial_start_idx_local),
                "trial_start_event_idx_global": int(trial_start_idx_global),
                "condition_code":  int(current_condition_code),
                "condition_label": current_condition_label,
                "reason": "missing ITI trigger",
            })
        else:
            candidate_trials.append({
                "condition_code":           int(current_condition_code),
                "condition_label":          current_condition_label,
                "condition_rep":            int(condition_rep[current_condition_code]),
                "trial_within_condition":   trial_within_condition,
                "trial_start_sample":       trial_start_sample,
                "iti_start_sample":         int(iti_sample),
                "trial_end_sample":         trial_end_sample,
                "trial_start_event_idx_local":  int(trial_start_idx_local),
                "trial_start_event_idx_global": int(trial_start_idx_global),
                "iti_event_idx_local":          int(iti_idx_local),
                "iti_event_idx_global":         int(iti_idx_global),
            })
            trial_within_condition += 1

        i = trial_start_idx_local + 1

    # --- Group-level validation (analogous to n_words in langloc) ---
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for trial in candidate_trials:
        key = (trial["condition_code"], trial["condition_rep"])
        groups[key].append(trial)

    valid_trials: list[dict] = []
    for group in groups.values():
        if len(group) == trials_per_condition:
            valid_trials.extend(group)
        else:
            for t in group:
                bad_trials.append({
                    "block_start_event_idx":        int(block_start_event_idx),
                    "trial_start_event_idx_local":  t["trial_start_event_idx_local"],
                    "trial_start_event_idx_global": t["trial_start_event_idx_global"],
                    "condition_code":               t["condition_code"],
                    "condition_label":              t["condition_label"],
                    "condition_rep":                t["condition_rep"],
                    "found_n_trials":               len(group),
                    "expected_n_trials":            trials_per_condition,
                    "reason": "wrong trial count in condition group",
                })

    # Restore temporal order and assign final trial_number
    valid_trials.sort(key=lambda t: t["trial_start_sample"])
    for n, t in enumerate(valid_trials):
        t["trial_number"] = n

    return valid_trials, bad_trials

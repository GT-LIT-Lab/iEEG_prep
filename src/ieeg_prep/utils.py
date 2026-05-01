"""Utilities for iEEG preprocessing."""

import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd



def label_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Set channel types on an MNE Raw object.

    Channel IDs are derived by stripping trailing digits from channel names.
    Left-hemisphere (names starting with 'l') and right-hemisphere (names starting with 'r')
    are labeled as 'seeg'. Known non-iEEG channels (EKG, TRIG, Pleth, OSAT, PR) are labeled
    with their appropriate types. Other channels keep their original types.
    Modifies the object in place and returns it. No file I/O.

    Args:
        raw: MNE Raw object (e.g. from mne.io.read_raw_edf).

    Returns:
        The same Raw object with channel types updated.
    """
    ch_names = raw.ch_names
    ch_IDs = sorted(set(re.sub(r"\d+$", "", ch) for ch in ch_names))
    lh_channels = [ch for ch in ch_IDs if ch[0].lower() == "l"]
    rh_channels = [ch for ch in ch_IDs if ch[0].lower() == "r"]

    # Identify all iEEG electrodes (left and right hemisphere)
    eeg_ch_set = lh_channels + rh_channels

    # This identifies all the iEEG channels on each electrode
    eeg_channels = [
        ch for ch in ch_names if re.sub(r"\d+$", "", ch) in eeg_ch_set
    ]

    # Set the channel types to 'seeg' for all iEEG channels
    raw.set_channel_types({ch: "seeg" for ch in eeg_channels})

    # Set types for known non-iEEG channels
    channel_type_map = {
        "EKG": "ecg",
        "TRIG": "stim",
        "Pleth": "misc",
        "OSAT": "misc",
        "PR": "misc",
    }
    known_ch_types = {ch: t for ch, t in channel_type_map.items() if ch in ch_names}
    dc_ch_types = {ch: "misc" for ch in ch_names if re.match(r"^DC\d+$", ch)}
    if known_ch_types or dc_ch_types:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The unit for channel")
            raw.set_channel_types({**known_ch_types, **dc_ch_types})

    return raw


def add_coordinates(
    raw: mne.io.Raw,
    csv_path: Path | str,
    channel_col: str = "label",
    x_col: str = "mni_x",
    y_col: str = "mni_y",
    z_col: str = "mni_z",
    coord_frame: str = "mni_tal",
) -> mne.io.Raw:
    """Set MNI montage on a Raw object from a CSV of electrode coordinates.

    Normalizes CSV channel labels to match MNE naming conventions
    (e.g. 'LAm-01' -> 'LAm1'). Assumes raw channel names are already
    correctly formatted. Modifies the object in place and returns it.
    Issues a warning for any SEEG channels that do not receive coordinates.

    Args:
        raw: MNE Raw object with channel types already set (e.g. after
            calling label_channels).
        csv_path: Path to CSV file containing electrode coordinates.
        channel_col: Column name for channel labels in the CSV.
        x_col: Column name for MNI x coordinates.
        y_col: Column name for MNI y coordinates.
        z_col: Column name for MNI z coordinates.
        coord_frame: Coordinate frame for the montage (default 'mni_tal').

    Returns:
        The same Raw object with the montage set.

    Raises:
        ValueError: If required columns are missing from the CSV, or if no
            channels overlap between the CSV and raw.ch_names.
    """
    df = pd.read_csv(csv_path)

    required_cols = [channel_col, x_col, y_col, z_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df = df[[channel_col, x_col, y_col, z_col]].dropna().copy()
    df[channel_col] = df[channel_col].astype(str)

    def _normalize_label(label: str) -> str:
        m = re.match(r"([A-Za-z]+)-?0*([0-9]+)$", label)
        if m:
            return f"{m.group(1)}{int(m.group(2))}"
        return label

    df["normalized_label"] = df[channel_col].apply(_normalize_label)
    csv_chs = set(df["normalized_label"])

    # Warn about SEEG channels that will not receive coordinates
    seeg_picks = mne.pick_types(raw.info, seeg=True)
    seeg_chs = [raw.ch_names[i] for i in seeg_picks]
    missing_seeg = [ch for ch in seeg_chs if ch not in csv_chs]
    if missing_seeg:
        warnings.warn(
            f"{len(missing_seeg)} SEEG channel(s) did not receive coordinates: "
            f"{missing_seeg}",
            RuntimeWarning,
            stacklevel=2,
        )

    matched_df = df[df["normalized_label"].isin(set(raw.ch_names))].copy()
    if matched_df.empty:
        raise ValueError("No overlapping channels found between CSV and raw.ch_names.")

    ch_pos = {
        row["normalized_label"]: np.array(
            [row[x_col], row[y_col], row[z_col]], dtype=float
        )
        for _, row in matched_df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Fiducial point nasion not found")
        raw.set_montage(montage, on_missing="ignore")

    return raw

def rename_channels(raw: mne.io.Raw, channel_names: list[{"name": str, "new_name": str}]) -> mne.io.Raw:
    """Rename channels in a Raw object. Sometimes the channel names are labeled incorrectly and need to be manually corrected. Example: RPul (lower case "L") in EMOP0004 is incorrectly labeled as RPuI (upper case "i")

    Args:
        raw: MNE Raw object.
        channel_names: List of dictionaries with "name" and "new_name" keys.
            channel_names should only include strings associated with depth electrodes and not
            individual channel numbers. e.g. "LAm" instead of "LAm1".

    Returns:
        The same Raw object with channels renamed.
    """

    for channel_name in channel_names:
        matched = [ch for ch in raw.ch_names if ch.startswith(channel_name["name"])]
        if not matched:
            warnings.warn(
                f"No channels found starting with '{channel_name['name']}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for ch in matched:
            new_ch_name = ch.replace(channel_name["name"], channel_name["new_name"])
            raw.rename_channels({ch: new_ch_name})

    return raw


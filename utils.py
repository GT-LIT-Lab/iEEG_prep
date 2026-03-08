"""Utilities for iEEG preprocessing."""

import re
from pathlib import Path

import mne



def label_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Set channel types for L/R channels to 'seeg' on an MNE Raw object.

    Channel IDs are derived by stripping trailing digits from channel names.
    Left-hemisphere (names starting with 'l') and right-hemisphere (names starting with 'r')
    are labeled as 'seeg'; other channels keep their original types.
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

    return raw



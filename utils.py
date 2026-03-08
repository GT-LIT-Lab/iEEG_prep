"""Utilities for iEEG preprocessing."""

import re
from pathlib import Path
import argparse

import mne



def label_channels_and_save_edf(edf_path: str | Path) -> None:
    """Load a raw EDF, set channel types for L/R channels to 'seeg', and overwrite the file.

    Channel IDs are derived by stripping trailing digits from channel names.
    Left-hemisphere (names starting with 'l') and right-hemisphere (names starting with 'r')
    are labeled as 'seeg'; other channels keep their original types.

    Args:
        edf_path: Path to the raw EDF file. The file is overwritten in place.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    ch_names = raw.ch_names
    ch_IDs = sorted(set(re.sub(r"\d+$", "", ch) for ch in ch_names))
    lh_channels = [ch for ch in ch_IDs if ch[0].lower() == "l"]
    rh_channels = [ch for ch in ch_IDs if ch[0].lower() == "r"]
    eeg_ch_set = lh_channels + rh_channels

    eeg_channels = [
        ch for ch in ch_names if re.sub(r"\d+$", "", ch) in eeg_ch_set
    ]
    raw.set_channel_types({ch: "seeg" for ch in eeg_channels})

    raw.save(edf_path, overwrite=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Set L/R channel types to 'seeg' and overwrite EDF in place."
    )
    parser.add_argument(
        "edf_path",
        type=Path,
        help="Path to the raw EDF file",
    )
    args = parser.parse_args()
    label_channels_and_save_edf(args.edf_path)
    print(f"Done: {args.edf_path}")
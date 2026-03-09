# iEEG Prep

LIT Lab iEEG preprocessing pipeline for intracranial EEG data.

Preprocessing methodology is based on [Regev and Casto et al. (2024)](https://doi.org/10.1038/s41562-024-01944-2).

---

## Overview

This package provides tools for:

- **Channel labeling** — automatically set channel types (sEEG, ECG, stimulus, misc) from naming conventions
- **Coordinate mapping** — add MNI electrode coordinates from a CSV to an MNE Raw object
- **IED detection** — detect interictal epileptiform discharges from sEEG recordings

## Quick install

```bash
pip install -e .
```

## Basic usage

```python
import mne
from iEEG_prep import label_channels, add_coordinates

raw = mne.io.read_raw_edf("recording.edf", preload=True)

label_channels(raw)
add_coordinates(raw, "electrodes.csv")
```

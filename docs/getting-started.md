# Getting Started

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd iEEG_prep
pip install -e .
```

## Typical preprocessing workflow

```python
import mne
from iEEG_prep import label_channels, add_coordinates

# 1. Load raw EDF
raw = mne.io.read_raw_edf("recording.edf", preload=True)

# 2. (Optional) Fix any incorrectly named electrodes before labeling
from iEEG_prep import utils
utils.rename_channels(raw, [{"name": "RPuI", "new_name": "RPul"}])

# 3. Label channel types
label_channels(raw)

# 4. Add MNI coordinates
add_coordinates(raw, "electrodes.csv")

# 5. Resample
raw.resample(400)

# 6. High-pass filter
raw.filter(l_freq=0.5, h_freq=None, picks="seeg", phase="zero")

# 7. Notch filter (line noise + harmonics)
for f in [60, 120, 180, 240]:
    raw.notch_filter(freqs=f, method="iir", picks="seeg")
```

## Running IED detection

```bash
python -m IED.IED_analysis --raw-path recording.edf --output-dir ./results
```

Or with a config file:

```bash
python -m IED.IED_analysis --config my_config.json
```

See the [IED Pipeline](INTERICTAL_PIPELINE_STEPS.md) page for full details.

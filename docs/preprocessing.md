# Preprocessing Guide

## Channel labeling

`label_channels` automatically assigns MNE channel types based on naming conventions:

| Pattern | Type assigned |
|---------|--------------|
| Starts with `L` or `R` + digits | `seeg` |
| `EKG` | `ecg` |
| `TRIG` | `stim` |
| `DC1`, `DC2`, ... | `misc` |
| `Pleth`, `OSAT`, `PR` | `misc` |

```python
from iEEG_prep import label_channels

label_channels(raw)
```

Call this before `add_coordinates` so that SEEG channels are identified correctly for the coordinate warning.

---

## Renaming channels

If channel names in the EDF are incorrect, fix them before calling `label_channels`:

```python
from iEEG_prep import utils

utils.rename_channels(raw, [
    {"name": "RPuI", "new_name": "RPul"},
])
```

`channel_names` entries match by prefix — `"RPuI"` will rename `RPuI1`, `RPuI2`, etc. A warning is issued for any entry that matches no channels.

---

## Adding MNI coordinates

`add_coordinates` reads a CSV of electrode positions and sets an MNI montage on the Raw object:

```python
from iEEG_prep import add_coordinates

add_coordinates(raw, "electrodes.csv")
```

**Expected CSV columns** (defaults, all configurable):

| Column | Description |
|--------|-------------|
| `label` | Channel name |
| `mni_x` | MNI x coordinate |
| `mni_y` | MNI y coordinate |
| `mni_z` | MNI z coordinate |

CSV labels are normalized automatically (e.g. `LAm-01` → `LAm1`), so the CSV does not need to match MNE naming exactly. A `RuntimeWarning` is issued for any SEEG channel that does not receive coordinates.

Custom column names:

```python
add_coordinates(
    raw,
    "electrodes.csv",
    channel_col="electrode",
    x_col="x",
    y_col="y",
    z_col="z",
)
```

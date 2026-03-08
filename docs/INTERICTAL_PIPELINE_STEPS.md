# Interictal (IED) pipeline – user guide

This document describes the **interictal epileptiform discharge (IED) detection pipeline**: how to run it, what it does, and what it outputs. The method is based on Janca et al. (2015).

**Reference:** Janca, R., et al. (2015). Detection of interictal epileptiform discharges using signal envelope distribution modelling. *Brain Topography*, 28(6), 770–778. https://link.springer.com/article/10.1007/s10548-014-0379-1

---

## Quick start

**Requirements:** EDF file with channel names that start with `L` or `R` (plus optional digits) for sEEG channels. Other channels (EKG, DC, etc.) are kept but not used for IED detection.

### Option 1: Config file

1. Copy the example config and edit paths and options:
   ```bash
   cp batch_script/ied_config_example.json my_config.json
   # Edit my_config.json: set raw_path and output_dir
   ```

2. Run the pipeline:
   ```bash
   python -m preprocessing.IED_analysis --config my_config.json
   ```

### Option 2: Command-line only

```bash
python -m preprocessing.IED_analysis \
  --raw-path /path/to/your/file.edf \
  --output-dir /path/to/output
```

### Option 3: HPC (SLURM)

```bash
sbatch batch_script/run_ied_analysis.sbatch --config my_config.json
# or
sbatch batch_script/run_ied_analysis.sbatch --raw-path file.edf --output-dir ./out
```

---

## Config file format

`config.json` can contain:

| Key | Required | Description |
|-----|----------|-------------|
| `raw_path` | **Yes** (if no `--raw-path`) | Path to the raw EDF file |
| `output_dir` | No (default: current directory) | Directory where results are saved |
| `channels_above_rate_thresh` | No (default: 6.5) | Spike rate threshold (spikes/min) for the `channels_above_rate` boolean |
| `interictal_config` | No | Object with pipeline parameters (see below) |

Any option can be overridden from the command line. Example config:

```json
{
  "raw_path": "/path/to/file.edf",
  "output_dir": "/path/to/output",
  "channels_above_rate_thresh": 6.5,
  "interictal_config": {
    "sfreq_target": 200,
    "bandpass_low": 10,
    "bandpass_high": 60,
    "notch_method": "custom_biquad",
    "notch_freq": 60,
    "win_coeff": 5,
    "overlap_coeff": 4,
    "threshold_k": 3.65,
    "polyspike_union_sec": 0.12,
    "trim_edges_sec": 1
  }
}
```

### Pipeline parameters (`interictal_config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sfreq_target` | 200 | Resampling target (Hz) |
| `bandpass_low` | 10 | Bandpass low cutoff (Hz) |
| `bandpass_high` | 60 | Bandpass high cutoff (Hz) |
| `bandpass_order` | 8 | IIR filter order |
| `bandpass_ftype` | `"cheby2"` | IIR type (`"cheby2"`, etc.) |
| `bandpass_rs` | 60 | Cheby2 stopband ripple (dB) |
| `notch_freq` | 60 | Line-noise notch frequency (Hz) |
| `notch_radius` | 0.985 | Pole radius for custom biquad notch |
| `notch_method` | `"custom_biquad"` | `"custom_biquad"` or `"mne"` |
| `win_coeff` | 5 | Envelope segment window (seconds) |
| `overlap_coeff` | 4 | Envelope segment overlap (seconds) |
| `smooth_window_len` | 5 | Moving-average length for μ/σ smoothing |
| `threshold_k` | 3.65 | Threshold multiplier: k × (mode + median) |
| `polyspike_union_sec` | 0.12 | Merge spikes within this window (seconds) |
| `trim_edges_sec` | 1 | Drop first/last N seconds |

---

## Outputs

All files are written under `output_dir`.

### `ied_results.npz`

NumPy archive (load with `np.load(...)`):

| Key | Shape / type | Description |
|-----|----------------|-------------|
| `spike_markers` | `(n_channels, n_samples)` bool | 1 where a spike was detected |
| `channel_spike_counts` | `(n_channels,)` int | Total spikes per channel |
| `channel_spike_rates` | `(n_channels,)` float | Spikes per minute per channel |
| `channels_above_rate` | `(n_channels,)` bool | Channels with rate > `channels_above_rate_thresh` |
| `ch_names` | 1D array of str | sEEG channel names (same order as above) |
| `sfreq` | scalar | Sampling frequency (Hz) after resampling |

### `ied_metadata.json`

Summary and run config:

- `sfreq`, `time_min`, `n_spikes_total`, `n_channels`, `n_samples`
- `ch_names`, `channels_above_rate_thresh`, `n_channels_above_rate`
- `config`: full `interictal_config` used for the run

### Optional (with `--return-intermediates`)

- `threshold.npy` – sample-wise threshold curve `(n_channels, n_samples)`
- `mu_interp.npy`, `sigma_interp.npy` – interpolated lognormal parameters

---

## Pipeline steps (what the code does)

1. **Load EDF** and set sEEG channel types (L/R channels → `seeg` via `utils.label_channels`).
2. **Resample** to `sfreq_target` (default 200 Hz).
3. **Bandpass** 10–60 Hz (Cheby2 IIR, order 8) on sEEG only.
4. **Notch** 60 Hz line noise (custom biquad or MNE IIR).
5. **Hilbert envelope** of the filtered sEEG.
6. **Lognormal fit** per sliding segment (window `win_coeff` s, overlap `overlap_coeff` s) → μ, σ per segment.
7. **Smooth** μ and σ along segments (moving average).
8. **Interpolate** μ and σ to every sample (cubic spline).
9. **Threshold** = k × (lognormal mode + lognormal median).
10. **Detection**: envelope > threshold → crossings → one peak per crossing → merge nearby spikes (polyspike union) → one local max per merged event → trim first/last N seconds.
11. **Rates**: spike count and rate per channel; boolean `channels_above_rate` using the chosen rate threshold.

---

## Using results in Python

```python
import numpy as np
import json

# Load arrays
data = np.load("output_dir/ied_results.npz")
spike_markers = data["spike_markers"]           # (n_channels, n_samples) bool
channel_spike_rates = data["channel_spike_rates"]
channels_above_rate = data["channels_above_rate"]
ch_names = data["ch_names"]

# Channels with high spike rate
high_rate_channels = ch_names[channels_above_rate]

# Load metadata
with open("output_dir/ied_metadata.json") as f:
    meta = json.load(f)
print(meta["n_spikes_total"], meta["time_min"], meta["channels_above_rate_thresh"])
```

---

## Command-line reference

```text
python -m preprocessing.IED_analysis [OPTIONS]
```

**Required (if not in config):** `--raw-path` path to EDF.

**Common options:**

- `--config PATH` – load options from JSON (must contain `raw_path` or use `--raw-path`)
- `--raw-path PATH` – path to raw EDF
- `--output-dir PATH` – where to write results (default: current directory)
- `--channels-above-rate-thresh FLOAT` – rate threshold for boolean mask (default: 6.5)
- `--return-intermediates` – also save threshold, mu_interp, sigma_interp
- `--notch-method {custom_biquad,mne}` – notch filter method
- `--sfreq-target`, `--bandpass-low`, `--bandpass-high`, etc. – override config

Run with `--help` for the full list.

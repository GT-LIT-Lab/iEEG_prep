# LIT LAB iEEG preprocessing pipeline

Preprocessing methodology in this repo is based on Regev and Casto et al. (2024).

- Regev, T.I., Casto, C., Hosseini, E.A. et al. Neural populations in the language network differ in the size of their temporal receptive windows. *Nat Hum Behav* 8, 1924–1942 (2024). https://doi.org/10.1038/s41562-024-01944-2

---

## Interictal (IED) pipeline

Detection of **interictal epileptiform discharges** from sEEG EDF files (Janca et al., 2015). The pipeline: loads raw EDF → labels L/R channels as sEEG → resamples → bandpass + notch → Hilbert envelope → lognormal threshold → spike detection → per-channel counts/rates and a boolean mask of high-rate channels.

**Quick start:**

```bash
# With a config file (copy batch_script/ied_config_example.json and set raw_path, output_dir)
python -m preprocessing.IED_analysis --config my_config.json

# Or with CLI args only
python -m preprocessing.IED_analysis --raw-path /path/to/file.edf --output-dir /path/to/output
```

**Outputs** (in `output_dir`): `ied_results.npz` (spike markers, channel counts/rates, `channels_above_rate` boolean, channel names), `ied_metadata.json` (summary + config). Optional intermediates with `--return-intermediates`.

**Full documentation:** [docs/INTERICTAL_PIPELINE_STEPS.md](docs/INTERICTAL_PIPELINE_STEPS.md) — config format, all parameters, output descriptions, HPC/SLURM usage, and example Python code to load results.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `mne`, `numpy`, `scipy`, and optionally `jaxtyping` for shape annotations in the interictal module.
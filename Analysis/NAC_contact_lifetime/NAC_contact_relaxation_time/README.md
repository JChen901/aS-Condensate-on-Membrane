# NAC contact relaxation and ACF

This folder calculates inter-chain backbone contacts for the original NAC,
N-terminal, and C-terminal contact analyses. It keeps the original output
layout and plotting workflow while using faster shared scans by default.

## Main files

- `nac_relaxation_time.ipynb`: notebook launcher for running calculations and
  plotting cached results.
- `nac_triplet_relaxation.py`: contact lifetime/survival analysis.
- `nac_triplet_acf.py`: contact autocorrelation analysis.
- `nac_plotting.py`: plotting helpers used by the notebook.

## Contact modes

The original NAC modes are unchanged:

- `any_nac_to_nac_bb`: any inter-chain NAC BB-BB contact.
- `corresponding_nac_bb`: at least one same-index NAC BB-BB contact.
- `three_corresponding_nac_bb`: at least three same-index NAC BB-BB contacts.
- `triplet_corresponding_nac_bb`: at least three consecutive same-index NAC
  BB-BB contacts.

The same scripts are also used by the notebook for N-terminal and C-terminal
contact ranges.

## Run survival/lifetime analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_relaxation.py
```

Default NAC outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_contact_outputs/
```

## Run ACF analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_acf.py
```

Default NAC ACF outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_contact_acf_outputs/
```

Use `--group` to run selected trajectory groups, `--mode` to run selected
contact modes, and `--force` to ignore existing CSV caches.

## Speedups

- Multiple contact modes share one trajectory scan by default.
- When `any_nac_to_nac_bb` is included, one cutoff-neighbor search per frame is
  reused to derive the corresponding-contact modes.
- The same corresponding-residue contact matrix is reused for `>=1`, `>=3`,
  and `3 consecutive` modes.
- Largest-cluster valid-chain masks can be cached on disk and reused when the
  same trajectory and filter settings are run again.
- ACF stores boolean contact-series arrays in memory, trading memory for fewer
  repeated trajectory reads and distance calculations.

The time interval is unchanged by default: `--skip-frame 1`.

Fallback options:

```bash
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_relaxation.py --separate-modes
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_acf.py --separate-modes
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_relaxation.py --no-valid-chain-cache
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_acf.py --no-valid-chain-cache
```

## Generated files

The `*_contact_outputs/` and `*_contact_acf_outputs/` folders contain generated
CSV and figure outputs. Keep them if you want the notebook to load existing
data and plot directly.

`valid_chain_cache/` is optional. It only stores reusable largest-cluster masks
for speed. It can be deleted at any time; the next calculation will regenerate
it if valid-chain caching is enabled.

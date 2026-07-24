# NAC antisymmetric contact relaxation and ACF

This folder mirrors `Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time`, but replaces same-index
corresponding contacts with reverse-index antisymmetric contacts.

For the default NAC range 61-95, the antisymmetric BB pairs are:

- 61 with 95
- 62 with 94
- ...
- 95 with 61

## Contact modes

- `any_nac_to_nac_bb`: any inter-chain NAC BB-BB contact, kept as a baseline.
- `antisymmetric_nac_bb`: at least one reverse-index NAC BB contact.
- `three_antisymmetric_nac_bb`: at least three reverse-index NAC BB contacts.
- `triplet_antisymmetric_nac_bb`: at least three consecutive reverse-index NAC BB contacts.

## Run survival/lifetime analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_triplet_relaxation.py
```

Outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_contact_outputs/
```

## Run ACF analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_triplet_acf.py
```

Outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_contact_acf_outputs/
```

Use `--mode` to run selected contact modes, `--group` to run selected trajectory
groups, and `--force` to ignore cached CSV files.

## Speedups in this version

- Multiple contact modes use a shared trajectory scan by default.
- The same antisymmetric residue-contact matrix is reused for `>=1`, `>=3`,
  and `3 consecutive` antisymmetric modes.
- Valid-chain/largest-cluster masks are cached under `valid_chain_cache/` and
  reused across survival and ACF runs when inputs and filter settings match.
- ACF shared scan stores several boolean contact-series arrays at once. This
  uses more memory but avoids repeated trajectory reads and distance
  calculations.

Fallback options:

```bash
python Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_triplet_acf.py --separate-modes
python Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_triplet_relaxation.py --no-valid-chain-cache
```

The time interval is unchanged by default: `--skip-frame 1`.

## Notebook launcher

Open and run:

```text
Analysis/NAC_contact_lifetime/NAC_antisymmetric_contact_relaxation_time/nac_relaxation_time.ipynb
```

The notebook exposes the same optimized defaults:

- `SHARED_SCAN = True`
- `USE_VALID_CHAIN_CACHE = True`
- `SKIP_FRAME = 1`

Plotting is done by the survival and ACF cells in the notebook, matching the
original workflow.

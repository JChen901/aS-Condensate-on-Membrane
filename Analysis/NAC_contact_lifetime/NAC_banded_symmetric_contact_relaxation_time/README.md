# NAC banded symmetric contact relaxation and ACF

This folder mirrors `Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time`, but replaces same-index
corresponding contacts with banded symmetric contacts.

For residues `i` and `j`, a banded symmetric pair satisfies:

```text
abs(resid_i - resid_j) < window
```

By default, `window = 8`.

For the default NAC range 61-95, this means:

```text
abs(resid_i - resid_j) < 8
```

## Contact modes

- `any_nac_to_nac_bb`: any inter-chain NAC BB-BB contact, kept as a baseline.
- `symmetric_nac_bb`: at least one banded symmetric NAC BB contact.
- `three_symmetric_nac_bb`: at least three residues with banded symmetric NAC BB contacts.
- `triplet_symmetric_nac_bb`: three consecutive source-chain residues contacting
  three consecutive partner-chain residues in the same direction, with each
  residue pair satisfying the banded symmetric condition.

## Run survival/lifetime analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_triplet_relaxation.py
```

Outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_contact_outputs/
```

## Run ACF analysis

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_triplet_acf.py
```

Outputs are written to:

```text
Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_contact_acf_outputs/
```

Use `--mode` to run selected contact modes, `--group` to run selected trajectory
groups, and `--force` to ignore cached CSV files.

Old cached CSV files made before the partner-chain continuity requirement are
not reused; the scripts will recalculate them and write the updated definition
tag into new CSV outputs.

## Speedups in this version

- Multiple contact modes use a shared trajectory scan by default.
- Banded symmetric contacts use one cutoff-neighbor search per frame, then
  filter the close residue pairs by the band condition instead of calculating
  every allowed banded residue-pair distance.
- The banded symmetric residue-contact matrix is reused for `>=1` and `>=3`
  modes. The `3 consecutive` mode additionally keeps partner-residue identity
  so both chains must contain a consecutive three-residue segment.
- Valid-chain/largest-cluster masks are cached under `valid_chain_cache/` and
  reused across survival and ACF runs when inputs and filter settings match.
- ACF shared scan stores several boolean contact-series arrays at once. This
  uses more memory but avoids repeated trajectory reads and distance
  calculations.

Fallback options:

```bash
python Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_triplet_acf.py --separate-modes
python Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_triplet_relaxation.py --no-valid-chain-cache
```

The time interval is unchanged by default: `--skip-frame 1`.

## Notebook launcher

Open and run:

```text
Analysis/NAC_contact_lifetime/NAC_banded_symmetric_contact_relaxation_time/nac_relaxation_time.ipynb
```

The notebook exposes the same optimized defaults:

- `SHARED_SCAN = True`
- `USE_VALID_CHAIN_CACHE = True`
- `SKIP_FRAME = 1`

Plotting is done by the survival and ACF cells in the notebook, matching the
original workflow.

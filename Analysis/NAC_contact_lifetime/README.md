# NAC contact lifetime analysis

This folder contains the four NAC contact lifetime/ACF analysis variants as
real subfolders. Each subfolder keeps its own notebook, scripts, generated CSV
tables, and figures.

## Subfolders

- `NAC_contact_relaxation_time`: original same-index corresponding-contact
  analysis.
- `NAC_antisymmetric_contact_relaxation_time`: strict antisymmetric-contact
  analysis.
- `NAC_banded_antisymmetric_contact_relaxation_time`: banded antisymmetric
  analysis using `abs(resid_i + resid_j - target) < 8`.
- `NAC_banded_symmetric_contact_relaxation_time`: banded symmetric analysis
  using `abs(resid_i - resid_j) < 8`.

## Cross-analysis plotting

`analysis_sets.csv` is the intended entry point for future plots that compare
all four analyses. It records each analysis label, folder, contact definition,
and the standard NAC/N-ter/C-ter survival and ACF output directories.

The paths in `analysis_sets.csv` are relative to this folder.

## Running existing analyses

Run a subfolder exactly as before, with the new parent path:

```bash
conda activate mdanalysis
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_relaxation.py
python Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_triplet_acf.py
```

The notebooks in each subfolder also find their own scripts when opened from
the project root, from `Analysis/`, from this folder, or from the subfolder
itself.

`valid_chain_cache/` and `__pycache__/` are optional generated caches. They can
be deleted without changing results.

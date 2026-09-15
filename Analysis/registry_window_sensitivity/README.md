# Persistent largest-cluster registry-window sensitivity

Windows: 4, 6, 8, 10. Definitions: banded symmetric and banded antisymmetric;
contacts: at least one, at least three, and three consecutive residue pairs.
Distance cutoff remains 8 Å; the registry window is a residue-index tolerance.

A chain must belong to the cached largest cluster at every analyzed frame.
Chain selection is fixed before comparing registry windows. Contact runs touching
the trajectory boundaries are retained, matching the parent persistent analysis.
The antisymmetric consecutive-triplet definition uses reverse partner direction.

The top-level `data/` directory contains the complete generated CSV output from
the trajectory analysis, including all pair types needed to reproduce the filtered
IDP-IDP view. Figures are written to the top-level `figures/` directory and show
IDP50, Hel50, IDP50-Mem, and mixed-system IDP-IDP only (chains 11-50). Crossing
times are computed from the mean survival curve on the parent analysis's 1000 ps
grid. Internal `data/` tables encode window as a variant suffix.

Run with the mdanalysis environment:

```bash
python execute_all.py
```

The runner uses three workers and reads each trajectory once for all windows.
`export_results.py` validates every window-8 histogram and trajectory summary
against the existing parent results, checks coverage, then plots.

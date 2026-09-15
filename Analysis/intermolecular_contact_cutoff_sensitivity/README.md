# Intermolecular contact cutoff sensitivity

This standalone analysis reuses the existing dense-phase BB selection and chain assignment (50 chains x 140 BB beads), PBC minimum-image distances, and frame-wise largest connected component convention. It analyzes the last 10 us using frames 10000-20000 at 20 ns spacing, independently for `alhx_0`, `alhx_700` and their three replicates. Cutoffs are 6, 7, 8, 9, and 10 A. No original trajectories or existing results are modified.

Run from the project root with:

```bash
/home/jychen/anaconda3/envs/mdanalysis/bin/python Analysis/intermolecular_contact_cutoff_sensitivity/analyze.py
```

Outputs are `summary.csv`, `jaccard_summary.csv`, `Nmax_time_series.csv`, `neighbor_number_distribution.csv`, `neighbor_number_cutoff_summary.csv`, and one combined 600-dpi PNG/PDF figure. The figure contains the largest-cluster size trend (A) above five neighbor-number distributions (6–10 Å) in one row (B), with shared distribution axis labels and 26-point panel letters; Jaccard is saved as statistics only. Styling follows the first cell of `Analysis/inter-chain/Net_Dist.ipynb`, including the Arial font and 180 mm publication width. Trend error bars show SD across replicate means; distributions pool chain/frame observations as in the notebook. Intermediate per-replicate arrays are under `data/`.

The 8 A cutoff is kept as the reference for Jaccard overlap. Largest-cluster assignment remains stable over 6-10 A: replicate-averaged relative changes in mean Nmax from 8 A are <=0.03% for `alhx_0` and <=0.25% for `alhx_700`, with mean Jaccard values of 0.998-1.000. Chain-level neighbor number is cutoff-sensitive in absolute magnitude: replicate-averaged values change from 9.06 -> 10.98 -> 13.19 -> 14.89 -> 16.13 for `alhx_0` and 6.26 -> 8.32 -> 10.59 -> 12.40 -> 13.74 for `alhx_700` at 6 -> 7 -> 8 -> 9 -> 10 A. The system ordering remains unchanged for both Nmax and mean neighbor number, but the absolute neighbor observable should not be described as cutoff-robust.

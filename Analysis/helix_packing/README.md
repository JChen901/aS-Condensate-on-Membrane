# NAC helix-packing analysis (band8)

This is the reduced, final workflow for testing NAC axial alignment and the
broad anti-diagonal contact pattern in the 50-chain solution simulations.
Only the strict <8-residue contact band is retained; residue offsets equal to
8 are excluded.

## Final directory structure

```text
helix_packing/
├── analyze_band8.py          # trajectory analysis only
├── plot_band8.ipynb          # plotting only
├── datasets.csv              # IDP50 and Hel50, three trajectories each
├── README.md                 # methods and usage
├── RESULTS.md                # numerical interpretation
└── results/
    ├── analysis_config.json
    ├── replicate_bin_metrics.csv
    ├── summary_bin_metrics.csv
    ├── cos_histogram.csv
    └── figures/
        ├── 01_signed_cos_distribution.{png,pdf,svg}
        ├── 02_axially_aligned_pairs.{png,pdf,svg}
        └── 03_anti_diagonal_preference.{png,pdf,svg}
```

## Run the numerical analysis

The analysis and plotting are deliberately independent. Run the trajectory
calculation with the MDAnalysis environment:

```bash
/home/jychen/anaconda3/envs/mdanalysis/bin/python \
  Analysis/helix_packing/analyze_band8.py
```

The default production setting samples the complete 0–20 microsecond
trajectories every 10 frames (10 ns). The script writes compact CSV tables but
does not import Matplotlib or create figures.

The six trajectories are configured in `datasets.csv`. Validate and list them
without reading trajectory frames using:

```bash
/home/jychen/anaconda3/envs/mdanalysis/bin/python \
  Analysis/helix_packing/analyze_band8.py --list-datasets
```

Run only one condition with `--dataset IDP50` or `--dataset Hel50`. The option
can be repeated when a custom manifest contains additional conditions.

## Draw the figures

Open `Analysis/helix_packing/plot_band8.ipynb` with the `CJY` kernel and run
all cells. The notebook:

- reads only the compact CSV results;
- explicitly loads `/data/jychen/package/Arial/arial.ttf`;
- creates separate 90 mm × 75 mm panels;
- saves 600-dpi PNG, PDF, and SVG files;
- contains the metric definitions beside the corresponding figures.

No bash wrapper is required.

If `results/` has been deleted, rerun `analyze_band8.py` first. The plotting
notebook does not recompute trajectory metrics; it only reads the compact CSV
and JSON files written under `results/`.

## System and contact definitions

- `IDP50`: `alhx_0`, three trajectories.
- `Hel50`: `alhx_700`, three trajectories.
- Each chain contains 140 BB beads.
- NAC is the within-chain residue range 61–95, inclusive.
- Two chains are connected in the cluster graph when any inter-chain BB pair
  is strictly closer than 8 Å. Cluster size is the connected-component size.
- Packing metrics only include chain pairs with a direct NAC–NAC contact
  strictly closer than 8 Å that belong to the same <8 Å connected component.
- Smooth contact strength is
  `q(d) = 0.5 * [1 - tanh(1 Å^-1 * (d - 8 Å))]`.
- The anti-diagonal band is `|r_i + r_j - 156| < 8`; values equal to 8 are
  excluded.
- The parallel band is `|r_i - r_j| < 8`; values equal to 8 are excluded.
- Cluster-size bins are `2–5`, `6–10`, `11–20`, `21–35`, and `36–50`.

## Final metrics

### Signed PCA-axis cos distribution

The interval `[-1, 1]` is divided into 40 bins of width 0.05. Counts are
normalized within each independent trajectory to obtain the probability in
each bin. The figure shows the mean probability per bin and its SEM across
trajectories. This is not a probability density, so the y-axis is directly
reported as a percentage per bin.

### Axially aligned pairs (%)

For each directly NAC-contacting chain pair, the principal axis of NAC BB
coordinates is calculated by PCA. Its sign is oriented N-to-C using the mean
positions of the first and last four NAC beads. A pair is called axially
aligned when

```text
|cos(theta)| >= 0.5
```

The plotted value is the percentage of directly contacting pairs satisfying
this condition.

### Antiparallel share (%)

This diagnostic remains in the CSV table but is not promoted to a main figure:

```text
100 * N(cos(theta) <= -0.5)
    / [N(cos(theta) <= -0.5) + N(cos(theta) >= 0.5)]
```

50% means no directional preference among aligned pairs.

### Anti-diagonal preference (%)

```text
100 * (mean(q_anti) / mean(q_parallel) - 1)
```

Positive values mean that the broad anti-diagonal band is stronger than the
parallel band; negative values indicate the opposite.

## Statistical unit

All metrics are calculated within each independent trajectory first. The
reported mean and SEM are then calculated across the three trajectory-level
values. Individual frames and chain-pair observations are not treated as
independent replicates.

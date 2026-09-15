# Persistent largest-cluster KS tests

This directory contains the KS analyses for the persistent largest-cluster
dataset. The parent `persistent_largest_cluster` directory generates the data and
non-KS figures; this directory reads those data files and writes KS-only outputs.

- `run_pooled_ks_test.py`: pooled pairwise KS tests across the four main systems.
- `four_main_systems/tables/`: pooled KS tables.
- `four_main_systems/figures/`: pooled KS heatmaps.
- `four_main_systems/replicate_ks_three_consecutive/`: replicate-resolved KS
  tests and ECDF plots for three-consecutive contacts.

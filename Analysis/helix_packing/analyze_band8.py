#!/usr/bin/env python3
"""Compute three intuitive NAC-packing metrics using a strict <8-residue band.

The script only writes compact replicate- and condition-level CSV tables. It
does not create figures; plotting is kept in ``plot_band8.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import (
    distance_array,
    minimize_vectors,
    self_capped_distance,
)


ATOMS_PER_CHAIN = 140
NAC_START = 61
NAC_END = 95
CLUSTER_CUTOFF = 8.0  # Angstrom
NAC_CONTACT_CUTOFF = 8.0  # Angstrom
CONTACT_D0 = 8.0  # Angstrom
BETA = 1.0  # Angstrom^-1
BANDWIDTH = 8  # residues; strict comparison, equality is excluded.
BAND_COMPARISON = "<"
END_WINDOW = 4
ALIGNMENT_THRESHOLD = 0.5
COS_EDGES = np.linspace(-1.0, 1.0, 41)

SIZE_BINS = (
    (2, 5, "2-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 35, "21-35"),
    (36, 50, "36-50"),
)
MANIFEST_COLUMNS = ("dataset", "replicate", "structure", "trajectory")


@dataclass
class Accumulator:
    n_pairs: int = 0
    n_antiparallel: int = 0
    n_parallel: int = 0
    anti_sum: float = 0.0
    parallel_sum: float = 0.0

    def add(self, cos_theta: float, anti_mean: float, parallel_mean: float) -> None:
        self.n_pairs += 1
        self.n_antiparallel += int(cos_theta <= -ALIGNMENT_THRESHOLD)
        self.n_parallel += int(cos_theta >= ALIGNMENT_THRESHOLD)
        self.anti_sum += anti_mean
        self.parallel_sum += parallel_mean

    def metrics(self) -> dict[str, float | int]:
        n_aligned = self.n_antiparallel + self.n_parallel
        anti_mean = self.anti_sum / self.n_pairs
        parallel_mean = self.parallel_sum / self.n_pairs
        return {
            "n_pairs": self.n_pairs,
            "n_aligned": n_aligned,
            "n_antiparallel": self.n_antiparallel,
            "n_parallel": self.n_parallel,
            "aligned_fraction_pct": 100.0 * n_aligned / self.n_pairs,
            "antiparallel_share_pct": (
                100.0 * self.n_antiparallel / n_aligned
                if n_aligned
                else float("nan")
            ),
            "anti_diagonal_mean": anti_mean,
            "parallel_diagonal_mean": parallel_mean,
            "anti_diagonal_preference_pct": (
                100.0 * (anti_mean / parallel_mean - 1.0)
                if parallel_mean > 1e-15
                else float("nan")
            ),
        }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "datasets.csv")
    parser.add_argument("--project-root", type=Path, default=here.parents[1])
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name from datasets.csv; repeat to select multiple datasets",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Validate datasets.csv, print selected inputs, and exit",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> list[dict]:
    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest}\n"
            "Expected CSV columns: " + ", ".join(MANIFEST_COLUMNS)
        )

    specs = []
    seen = set()
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{manifest} is missing columns: {sorted(missing_columns)}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = {column: (raw_row.get(column) or "").strip() for column in MANIFEST_COLUMNS}
            empty_columns = [column for column, value in row.items() if not value]
            if empty_columns:
                raise ValueError(
                    f"{manifest}:{line_number} has empty fields: {empty_columns}"
                )
            if args.dataset and row["dataset"] not in args.dataset:
                continue

            identity = (row["dataset"], row["replicate"])
            if identity in seen:
                raise ValueError(
                    f"Duplicate dataset/replicate in {manifest}: {identity}"
                )
            seen.add(identity)

            for key in ("structure", "trajectory"):
                path = Path(row[key]).expanduser()
                row[key] = path if path.is_absolute() else args.project_root / path
                if not row[key].exists():
                    raise FileNotFoundError(
                        f"{manifest}:{line_number} {key} not found: {row[key]}"
                    )
            specs.append(row)
    if not specs:
        requested = f" for --dataset {args.dataset}" if args.dataset else ""
        raise ValueError(f"No trajectory selected from {manifest}{requested}")
    return specs


def print_datasets(specs: list[dict]) -> None:
    print(f"Loaded {len(specs)} trajectories:")
    for spec in specs:
        print(
            f"  {spec['dataset']}/{spec['replicate']}: "
            f"{spec['trajectory']}"
        )


def size_bin(size: int) -> str:
    for lower, upper, label in SIZE_BINS:
        if lower <= size <= upper:
            return label
    raise ValueError(f"Cluster size outside configured bins: {size}")


def unwrap_segment(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    steps = minimize_vectors(np.diff(coords, axis=0), box)
    result = np.empty_like(coords, dtype=float)
    result[0] = coords[0]
    result[1:] = coords[0] + np.cumsum(steps, axis=0)
    return result


def directed_pca_axis(coords: np.ndarray) -> np.ndarray:
    ntoc = coords[-END_WINDOW:].mean(axis=0) - coords[:END_WINDOW].mean(axis=0)
    ntoc /= np.linalg.norm(ntoc)
    centered = coords - coords.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axis = eigenvectors[:, -1]
    if np.dot(axis, ntoc) < 0:
        axis = -axis
    return axis


def unique_chain_pairs(
    atom_pairs: np.ndarray, n_chains: int
) -> np.ndarray:
    if len(atom_pairs) == 0:
        return np.empty((0, 2), dtype=int)
    chains = atom_pairs // ATOMS_PER_CHAIN
    chains.sort(axis=1)
    chains = chains[chains[:, 0] != chains[:, 1]]
    if len(chains) == 0:
        return np.empty((0, 2), dtype=int)
    packed = np.unique(chains[:, 0] * n_chains + chains[:, 1])
    return np.column_stack((packed // n_chains, packed % n_chains)).astype(int)


def component_labels_and_sizes(
    n_chains: int, chain_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = [[] for _ in range(n_chains)]
    for i, j in chain_pairs:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    sizes = np.zeros(n_chains, dtype=int)
    labels = np.full(n_chains, -1, dtype=int)
    seen = np.zeros(n_chains, dtype=bool)
    for seed in range(n_chains):
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        members = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        sizes[members] = len(members)
        labels[members] = seed
    return labels, sizes


def contact_masks() -> tuple[np.ndarray, np.ndarray]:
    residues = np.arange(NAC_START, NAC_END + 1)
    ri, rj = np.meshgrid(residues, residues, indexing="ij")
    anti = np.abs(ri + rj - (NAC_START + NAC_END)) < BANDWIDTH
    parallel = np.abs(ri - rj) < BANDWIDTH
    return anti, parallel


def analyze_trajectory(
    spec: dict,
    args: argparse.Namespace,
    accumulators: dict[tuple[str, str, str], Accumulator],
    histograms: dict[tuple[str, str], np.ndarray],
) -> None:
    universe = mda.Universe(str(spec["structure"]), str(spec["trajectory"]))
    bb = universe.select_atoms("name BB")
    if len(bb) % ATOMS_PER_CHAIN:
        raise ValueError("Selected BB count is not divisible by 140")
    n_chains = len(bb) // ATOMS_PER_CHAIN
    nac_slice = slice(NAC_START - 1, NAC_END)
    nac_length = NAC_END - NAC_START + 1
    anti_mask, parallel_mask = contact_masks()
    stop = min(args.stop or len(universe.trajectory), len(universe.trajectory))
    frames = range(args.start, stop, args.stride)
    print(
        f"[{spec['dataset']}/{spec['replicate']}] "
        f"frames {args.start}:{stop}:{args.stride}",
        flush=True,
    )

    for processed, frame in enumerate(frames, start=1):
        ts = universe.trajectory[frame]
        coords = bb.positions.reshape(n_chains, ATOMS_PER_CHAIN, 3)
        atom_pairs, atom_distances = self_capped_distance(
            bb.positions,
            max_cutoff=max(CLUSTER_CUTOFF, NAC_CONTACT_CUTOFF),
            box=ts.dimensions,
            return_distances=True,
        )
        cluster_atom_pairs = atom_pairs[atom_distances < CLUSTER_CUTOFF]
        chain_pairs = unique_chain_pairs(cluster_atom_pairs, n_chains)
        component_labels, sizes = component_labels_and_sizes(n_chains, chain_pairs)

        local_i = atom_pairs[:, 0] % ATOMS_PER_CHAIN
        local_j = atom_pairs[:, 1] % ATOMS_PER_CHAIN
        nac_contact = (
            (atom_pairs[:, 0] // ATOMS_PER_CHAIN != atom_pairs[:, 1] // ATOMS_PER_CHAIN)
            & (atom_distances < NAC_CONTACT_CUTOFF)
            & (local_i >= nac_slice.start)
            & (local_i < nac_slice.stop)
            & (local_j >= nac_slice.start)
            & (local_j < nac_slice.stop)
        )
        nac_pairs = unique_chain_pairs(atom_pairs[nac_contact], n_chains)
        if len(nac_pairs):
            same_component = (
                component_labels[nac_pairs[:, 0]]
                == component_labels[nac_pairs[:, 1]]
            )
            nac_pairs = nac_pairs[same_component]
        participating = np.unique(nac_pairs) if len(nac_pairs) else np.empty(0, dtype=int)
        nac_coords: dict[int, np.ndarray] = {}
        axes: dict[int, np.ndarray] = {}
        for chain in participating:
            chain = int(chain)
            segment = unwrap_segment(coords[chain, nac_slice], ts.dimensions)
            if segment.shape != (nac_length, 3):
                raise RuntimeError("Unexpected NAC coordinate shape")
            nac_coords[chain] = segment
            axes[chain] = directed_pca_axis(segment)

        for i_raw, j_raw in nac_pairs:
            i, j = int(i_raw), int(j_raw)
            cos_theta = float(np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0))
            bin_index = int(np.searchsorted(COS_EDGES, cos_theta, side="right") - 1)
            bin_index = min(len(COS_EDGES) - 2, max(0, bin_index))
            histograms[(spec["dataset"], spec["replicate"])][bin_index] += 1
            distances = distance_array(nac_coords[i], nac_coords[j], box=ts.dimensions)
            contacts = 0.5 * (1.0 - np.tanh(BETA * (distances - CONTACT_D0)))
            anti_mean = float(np.mean(contacts[anti_mask]))
            parallel_mean = float(np.mean(contacts[parallel_mask]))
            label = size_bin(int(sizes[i]))
            key = (spec["dataset"], spec["replicate"], label)
            accumulators[key].add(cos_theta, anti_mean, parallel_mean)
            all_key = (spec["dataset"], spec["replicate"], "all")
            accumulators[all_key].add(cos_theta, anti_mean, parallel_mean)

        if args.progress_every and processed % args.progress_every == 0:
            print(
                f"  {processed} frames; frame={frame}; NAC pairs={len(nac_pairs)}",
                flush=True,
            )


def sem(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return (
        float(np.std(finite, ddof=1) / math.sqrt(len(finite)))
        if len(finite) > 1
        else float("nan")
    )


def write_tables(output_dir: Path, accumulators: dict, histograms: dict) -> None:
    metric_names = (
        "aligned_fraction_pct",
        "antiparallel_share_pct",
        "anti_diagonal_mean",
        "parallel_diagonal_mean",
        "anti_diagonal_preference_pct",
    )
    replicate_rows = []
    order = {label: index for index, (_, _, label) in enumerate(SIZE_BINS)}
    order["all"] = len(order)
    for (dataset, replicate, label), accumulator in sorted(
        accumulators.items(),
        key=lambda item: (item[0][0], item[0][1], order[item[0][2]]),
    ):
        replicate_rows.append(
            {
                "dataset": dataset,
                "replicate": replicate,
                "cluster_size_bin": label,
                **accumulator.metrics(),
            }
        )

    replicate_path = output_dir / "replicate_bin_metrics.csv"
    with replicate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=replicate_rows[0].keys())
        writer.writeheader()
        writer.writerows(replicate_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in replicate_rows:
        grouped[(row["dataset"], row["cluster_size_bin"])].append(row)
    summary_rows = []
    for (dataset, label), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], order[item[0][1]])
    ):
        summary = {
            "dataset": dataset,
            "cluster_size_bin": label,
            "n_replicates": len(rows),
            "n_pairs": sum(int(row["n_pairs"]) for row in rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows])
            finite = values[np.isfinite(values)]
            summary[f"mean_{metric}"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            summary[f"sem_{metric}"] = sem(values)
        summary_rows.append(summary)

    summary_path = output_dir / "summary_bin_metrics.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {replicate_path}")
    print(f"Wrote {summary_path}")

    histogram_path = output_dir / "cos_histogram.csv"
    with histogram_path.open("w", newline="") as handle:
        fields = ("dataset", "replicate", "cos_left", "cos_right", "count")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (dataset, replicate), counts in sorted(histograms.items()):
            for index, count in enumerate(counts):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "replicate": replicate,
                        "cos_left": COS_EDGES[index],
                        "cos_right": COS_EDGES[index + 1],
                        "count": int(count),
                    }
                )
    print(f"Wrote {histogram_path}")


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.start < 0:
        raise ValueError("stride must be positive and start non-negative")
    specs = load_manifest(args)
    print_datasets(specs)
    if args.list_datasets:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[tuple[str, str, str], Accumulator] = defaultdict(Accumulator)
    histograms: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(len(COS_EDGES) - 1, dtype=np.int64)
    )
    for spec in specs:
        analyze_trajectory(spec, args, accumulators, histograms)
    write_tables(args.output_dir, accumulators, histograms)

    config = {
        "analysis": "NAC helix packing, band8 only",
        "nac_residues": [NAC_START, NAC_END],
        "atoms_per_chain": ATOMS_PER_CHAIN,
        "cluster_cutoff_A": CLUSTER_CUTOFF,
        "nac_contact_cutoff_A": NAC_CONTACT_CUTOFF,
        "contact_d0_A": CONTACT_D0,
        "beta_per_A": BETA,
        "anti_diagonal_bandwidth_residues": BANDWIDTH,
        "band_comparison": BAND_COMPARISON,
        "anti_diagonal_band_expression": (
            f"|r_i + r_j - {NAC_START + NAC_END}| {BAND_COMPARISON} {BANDWIDTH}"
        ),
        "parallel_band_expression": f"|r_i - r_j| {BAND_COMPARISON} {BANDWIDTH}",
        "axis": "signed PCA axis, oriented N-to-C using four terminal NAC beads",
        "alignment_threshold_abs_cos": ALIGNMENT_THRESHOLD,
        "cos_histogram_edges": COS_EDGES.tolist(),
        "cluster_size_bins": [list(item) for item in SIZE_BINS],
        "frame_start": args.start,
        "frame_stop": args.stop,
        "frame_stride": args.stride,
        "MDAnalysis_version": mda.__version__,
        "numpy_version": np.__version__,
        "trajectories": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in spec.items()
            }
            for spec in specs
        ],
    }
    with (args.output_dir / "analysis_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    print(f"Wrote {args.output_dir / 'analysis_config.json'}")


if __name__ == "__main__":
    main()

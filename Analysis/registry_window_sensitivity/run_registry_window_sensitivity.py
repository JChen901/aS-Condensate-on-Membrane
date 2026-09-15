"""Registry-window sensitivity for persistent-largest-cluster chains."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
COMPARISON = PARENT.parent
CURRENT = COMPARISON / "instantaneous_largest_cluster" / "main_analysis"
LIFETIME = COMPARISON.parent
WINDOWS = (4, 6, 8, 10)
VARIANTS = ("banded_sym", "banded_anti")
CONTACTS = ("one", "three", "triplet")
REPS = (1, 2, 3)
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "registry_window_sensitivity_mpl"))

import numpy as np
import pandas as pd


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("base_contacts", LIFETIME / "NAC_contact_relaxation_time" / "nac_triplet_relaxation.py")
band_sym = load("banded_symmetric", LIFETIME / "NAC_banded_symmetric_contact_relaxation_time" / "nac_triplet_relaxation.py")
band_anti = load("banded_antisymmetric", LIFETIME / "NAC_banded_antisymmetric_contact_relaxation_time" / "nac_triplet_relaxation.py")
MODULES = {"banded_sym": band_sym, "banded_anti": band_anti}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_lengths(counts: defaultdict, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    lengths, n = np.unique(values, return_counts=True)
    for length, count in zip(lengths, n):
        counts[int(length)] += int(count)


def update(active: np.ndarray, contact: np.ndarray, counts: defaultdict) -> None:
    ended = (active > 0) & ~contact
    add_lengths(counts, active[ended])
    active[contact] += 1
    active[ended] = 0


def pair_types(group_name: str, pairs: np.ndarray, persistent: np.ndarray) -> dict[str, np.ndarray]:
    eligible = persistent[pairs[:, 0]] & persistent[pairs[:, 1]]
    result = {"all": eligible}
    if group_name == "G4_Onmem_wall":
        result.update({
            "Hel-Hel": eligible & (pairs[:, 1] < 10),
            "Hel-IDP": eligible & (pairs[:, 0] < 10) & (pairs[:, 1] >= 10),
            "IDP-IDP": eligible & (pairs[:, 0] >= 10),
        })
    return result


def cache_path(group_name: str, rep: int) -> Path:
    return CURRENT / "data" / group_name / f"largest_cluster_membership_rep{rep}.npz"


def allowed_matrices(n_residues: int) -> dict[tuple[str, int], np.ndarray]:
    allowed = {}
    for variant, module in MODULES.items():
        for window in WINDOWS:
            if variant == "banded_sym":
                a, b = module.build_banded_symmetric_indices((61, 95), symmetric_window=window)
            else:
                a, b, _target = module.build_banded_antisymmetric_indices((61, 95), antisymmetric_sum_window=window)
            allowed[variant, window] = module.build_banded_allowed_matrix(n_residues, a, b)
    return allowed


def analyze_rep(group: dict, rep: int) -> None:
    name, trajectory = group["name"], group["trajectories"][rep - 1]
    out = ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_path(name, rep)
    membership = np.load(cache, allow_pickle=False)["membership"].astype(bool)
    metadata_path = cache.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    universe = base.mda.Universe(trajectory["tpr"], trajectory["xtc"])
    _bb, nac, n_chains = base.build_chain_bb_indices(universe)
    if n_chains != 50 or nac.shape != (50, 35):
        raise ValueError(f"Unexpected protein topology: {n_chains} chains, NAC shape {nac.shape}")
    frames = base.get_frame_iterator(universe, trajectory["start"], -1, 1)
    if len(frames) != len(membership):
        raise ValueError(f"Membership/frame mismatch for {name} rep{rep}: {len(membership)} vs {len(frames)}")
    persistent = membership.all(axis=0)
    pairs = base.build_chain_pairs(n_chains)
    pair_index = base.build_pair_index_matrix(n_chains, pairs)
    mapping = np.arange(len(pairs), dtype=np.int32)
    subsets = pair_types(name, pairs, persistent)
    if any(int(mask.sum()) == 0 for mask in subsets.values()):
        raise RuntimeError(f"No persistent eligible pairs for {name} rep{rep}: " + str({k: int(v.sum()) for k, v in subsets.items()}))
    flat_chain, flat_residue = base.build_flat_chain_residue_ids(n_chains, 35)
    allowed = allowed_matrices(35)
    keys = [(variant, window, contact, pair_type) for variant in VARIANTS for window in WINDOWS for contact in CONTACTS for pair_type in subsets]
    active = {key: np.zeros(int(subsets[key[3]].sum()), dtype=np.int32) for key in keys}
    counts = {key: defaultdict(int) for key in keys}
    dt_ps = float(universe.trajectory.dt)

    for frame, ts in enumerate(frames):
        positions = universe.atoms.positions[nac]
        close = base.capped_distance(
            positions.reshape(-1, 3),
            positions.reshape(-1, 3),
            max_cutoff=8.0,
            box=base.get_valid_box(ts, True),
            return_distances=False,
        )
        for variant, module in MODULES.items():
            for window in WINDOWS:
                residues, pair_contacts = module.banded_residue_contacts_from_close_pairs(
                    close,
                    flat_chain,
                    flat_residue,
                    pair_index,
                    mapping,
                    np.ones(n_chains, dtype=bool),
                    allowed[variant, window],
                    len(pairs),
                    35,
                    return_pair_contacts=True,
                )
                direction = "reverse" if variant == "banded_anti" else "forward"
                masks = module.build_banded_contact_masks(
                    residues,
                    module.DEFAULT_CONTACT_MODES,
                    33,
                    residue_pair_contacts=pair_contacts,
                    triplet_partner_direction=direction,
                )
                native_names = {
                    "one": "symmetric_nac_bb" if variant == "banded_sym" else "antisymmetric_nac_bb",
                    "three": "three_symmetric_nac_bb" if variant == "banded_sym" else "three_antisymmetric_nac_bb",
                    "triplet": "triplet_symmetric_nac_bb" if variant == "banded_sym" else "triplet_antisymmetric_nac_bb",
                }
                for contact, native_name in native_names.items():
                    full = masks[native_name]
                    for pair_type, subset in subsets.items():
                        update(active[variant, window, contact, pair_type], full[subset], counts[variant, window, contact, pair_type])
        if frame % 1000 == 0:
            print(f"{name} rep {rep}: {frame}/{len(membership)}", flush=True)

    rows, hist = [], []
    for key in keys:
        variant, window, contact, pair_type = key
        add_lengths(counts[key], active[key][active[key] > 0])
        result = base.counts_to_survival(counts[key], dt_ps)
        row = {
            "group": name,
            "replicate": rep,
            "variant": f"{variant}_w{window}",
            "contact": contact,
            "pair_type": pair_type,
            "dt_ps": dt_ps,
            "n_frames": len(membership),
            "n_persistent_chains": int(persistent.sum()),
            "n_candidate_pairs": int(subsets[pair_type].sum()),
            "n_events": result["n_events"],
            "mean_lifetime_ps": result["mean_lifetime_ps"],
            "median_lifetime_ps": result["median_lifetime_ps"],
            "max_lifetime_ps": result["max_lifetime_ps"],
            "t_01_ps": base.crossing_time(result["time_ps"], result["survival"], 0.1),
            "t_1e_ps": base.crossing_time(result["time_ps"], result["survival"], 1 / np.e),
        }
        rows.append(row)
        hist.extend({
            "group": name,
            "replicate": rep,
            "variant": f"{variant}_w{window}",
            "contact": contact,
            "pair_type": pair_type,
            "lifetime_frames": int(length),
            "count": int(number),
            "event_status": "persistent_chain",
        } for length, number in sorted(counts[key].items()))
    pd.DataFrame(rows).to_csv(out / f"trajectory_summary_rep{rep}.csv", index=False)
    pd.DataFrame(hist).to_csv(out / f"event_histograms_rep{rep}.csv", index=False)
    (out / f"provenance_rep{rep}.json").write_text(json.dumps({
        "definition": "chain belongs to largest cluster at every analysed frame",
        "membership_cache": str(cache),
        "membership_cache_sha256": sha256(cache),
        "membership_metadata": metadata,
        "trajectory": trajectory,
        "windows": WINDOWS,
    }, indent=2) + "\n")


def summarize_group(group: dict) -> None:
    out = ROOT / "data" / group["name"]
    trajectories = pd.concat([pd.read_csv(out / f"trajectory_summary_rep{rep}.csv") for rep in REPS], ignore_index=True)
    histograms = pd.concat([pd.read_csv(out / f"event_histograms_rep{rep}.csv") for rep in REPS], ignore_index=True)
    trajectories.to_csv(out / "trajectory_summary.csv", index=False)
    histograms.to_csv(out / "event_histograms.csv", index=False)
    summaries, curves = [], []
    for (variant, contact, pair_type), sub in trajectories.groupby(["variant", "contact", "pair_type"], sort=False):
        time_list, survival_list = [], []
        for rep in REPS:
            d = histograms[
                (histograms.replicate == rep)
                & (histograms.variant == variant)
                & (histograms.contact == contact)
                & (histograms.pair_type == pair_type)
            ]
            result = base.counts_to_survival(
                dict(zip(d.lifetime_frames.astype(int), d["count"].astype(int))),
                float(sub[sub.replicate == rep].dt_ps.iloc[0]),
            )
            if result["n_events"]:
                time_list.append(result["time_ps"])
                survival_list.append(result["survival"])
        max_time = max(x[-1] for x in time_list)
        grid = np.arange(int(max_time / 1000) + 1, dtype=float) * 1000.0
        stack = np.array([np.interp(grid, x, y, left=1.0, right=0.0) for x, y in zip(time_list, survival_list)])
        mean, std = stack.mean(axis=0), stack.std(axis=0)
        summaries.append({
            "variant": variant,
            "contact": contact,
            "pair_type": pair_type,
            "n_events": int(sub.n_events.sum()),
            "t_01_ps": base.crossing_time(grid, mean, 0.1),
            "t_1e_ps": base.crossing_time(grid, mean, 1 / np.e),
            "mean_lifetime_ps": float(sub.mean_lifetime_ps.mean()),
            "mean_lifetime_std_ps": float(sub.mean_lifetime_ps.std(ddof=0)),
        })
        curves.extend({
            "variant": variant,
            "contact": contact,
            "pair_type": pair_type,
            "time_ps": float(t),
            "survival_mean": float(m),
            "survival_std": float(s),
        } for t, m, s in zip(grid, mean, std))
    pd.DataFrame(summaries).to_csv(out / "lifetime_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "survival_curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, help="0-based system index")
    parser.add_argument("--rep", type=int, help="1-based trajectory index")
    parser.add_argument("--summarise", "--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarise:
        for group in base.DEFAULT_GROUPS:
            summarize_group(group)
        return
    if args.group is not None or args.rep is not None:
        if args.group is None or args.rep is None:
            parser.error("--group and --rep must be provided together")
        group = base.DEFAULT_GROUPS[args.group]
        analyze_rep(group, args.rep)
        return
    for group in base.DEFAULT_GROUPS:
        for rep in REPS:
            analyze_rep(group, rep)
        summarize_group(group)


if __name__ == "__main__":
    main()

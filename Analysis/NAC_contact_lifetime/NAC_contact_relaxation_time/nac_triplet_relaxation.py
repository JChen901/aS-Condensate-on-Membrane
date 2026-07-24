import argparse
import hashlib
import json
import os
from collections import defaultdict

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import calc_bonds, capped_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


NAC_RESIDUE_RANGE = (61, 95)
RESIDUES_PER_CHAIN = 140
NAC_TRIPLET_SIZE = 3
NAC_BACKBONE_NAME = "BB"
NAC_CONTACT_CUTOFF_NM = 0.8
NAC_PAIR_CHUNK = 20000
NAC_USE_PBC = True
NAC_USE_RADIUS_FILTER = True
NAC_CHAIN_FILTER_METHOD = "largest_cluster"
NAC_CLUSTER_DIST_CUTOFF_A = 8.0
NAC_MIN_LIFETIME_FRAMES = 1

NAC_CACHE_CSV = "nac_triplet_survival_data.csv"
NAC_SUMMARY_CSV = "nac_triplet_relaxation_summary.csv"
NAC_TRAJ_SUMMARY_CSV = "nac_triplet_trajectory_summary.csv"

DEFAULT_RADIUS_LIMIT_NM = 10.0
DEFAULT_END_FRAME = -1
DEFAULT_SKIP_FRAME = 1
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_CONTACT_MODES = (
    "any_nac_to_nac_bb",
    "three_corresponding_nac_bb",
    "corresponding_nac_bb",
    "triplet_corresponding_nac_bb",
)
DEFAULT_GROUP_CHAIN_SELECTIONS = {
    "G3_Onmem_plus": "last_40",
}

CONTACT_MODE_CONFIG = {
    "triplet_corresponding_nac_bb": {
        "label": "3 consecutive corresponding NAC BB",
        "observable": "NAC_triplet_corresponding_BB_contact_survival",
        "file_prefix": "nac_triplet_corresponding_bb",
        "event_unit": "chain_pair",
    },
    "corresponding_nac_bb": {
        "label": "Any 1 corresponding NAC BB",
        "observable": "NAC_corresponding_BB_contact_survival",
        "file_prefix": "nac_corresponding_bb",
        "event_unit": "chain_pair",
    },
    "three_corresponding_nac_bb": {
        "label": "Any 3 corresponding NAC BB",
        "observable": "NAC_three_corresponding_BB_contact_survival",
        "file_prefix": "nac_three_corresponding_bb",
        "event_unit": "chain_pair",
    },
    "any_nac_to_nac_bb": {
        "label": "Any NAC BB to NAC BB",
        "observable": "NAC_any_to_NAC_BB_contact_survival",
        "file_prefix": "nac_any_nac_to_nac_bb",
        "event_unit": "chain_pair",
    },
}

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_VALID_CHAIN_CACHE_DIR = os.path.join(SCRIPT_DIR, "valid_chain_cache")


def find_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, "Analysis"))
            and os.path.isdir(os.path.join(current, "W_cluster"))
            and os.path.isdir(os.path.join(current, "Onmem"))
        ):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                f"Cannot infer project root from {start_dir!r}; expected parent directories "
                "to contain Analysis/, W_cluster/, and Onmem/."
            )
        current = parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


DEFAULT_GROUPS = [
    {
        "name": "G1_Wcluster_alhx0",
        "label": r"$\text{IDP}_{50}$",
        "color": "#074166",
        "trajectories": [
            {
                "tpr": project_path("W_cluster", "alhx_0", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_0", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("W_cluster", "alhx_0_re1", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_0_re1", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("W_cluster", "alhx_0_re2", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_0_re2", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
        ],
    },
    {
        "name": "G2_Wcluster_alhx700",
        "label": r"$\text{Hel}_{50}$",
        "color": "#CC011F",
        "trajectories": [
            {
                "tpr": project_path("W_cluster", "alhx_700", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_700", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("W_cluster", "alhx_700_re1", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_700_re1", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("W_cluster", "alhx_700_re2", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("W_cluster", "alhx_700_re2", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
        ],
    },
    {
        "name": "G3_Onmem_plus",
        "label": r"$\text{IDP}_{50}$-Mem",
        "color": "#8CC5BE",
        "chain_selection": "last_40",
        "trajectories": [
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus1", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus1", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 0,
            },
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus2", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus2", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 0,
            },
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus3", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_0_plus3", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 0,
            },
        ],
    },
    {
        "name": "G4_Onmem_wall",
        "label": r"$\text{Hel}_{10}\text{IDP}_{40}$-Mem",
        "color": "#FADADD",
        "trajectories": [
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL1", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL1", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL2", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL2", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
            {
                "tpr": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL3", "TR_Mdvwhole", "pbc.tpr"),
                "xtc": project_path("Onmem", "muti_chain", "POPC7-POPG3", "alhx_700_WALL3", "TR_Mdvwhole", "clus_centered.xtc"),
                "start": 10000,
            },
        ],
    },
]


def crossing_time(time_axis, y, threshold):
    below = np.where(y <= threshold)[0]
    if len(below) == 0:
        return np.nan

    i = below[0]
    if i == 0:
        return float(time_axis[0])

    t0, t1 = time_axis[i - 1], time_axis[i]
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))

def get_contact_mode_config(contact_mode):
    if contact_mode not in CONTACT_MODE_CONFIG:
        choices = ", ".join(CONTACT_MODE_CONFIG)
        raise ValueError(f"Unknown contact_mode={contact_mode!r}. Choices: {choices}")
    return CONTACT_MODE_CONFIG[contact_mode]


def add_lengths_to_counts(counts, lengths):
    if len(lengths) == 0:
        return

    values, value_counts = np.unique(lengths, return_counts=True)
    for value, count in zip(values, value_counts):
        counts[int(value)] += int(count)


def update_active_lifetimes(active_lengths, contact_mask, lifetime_counts):
    ended = (active_lengths > 0) & (~contact_mask)
    add_lengths_to_counts(lifetime_counts, active_lengths[ended])

    active_lengths[contact_mask] += 1
    active_lengths[ended] = 0


def counts_to_survival(counts, dt_ps, min_lifetime_frames=1):
    filtered = {
        int(length): int(count)
        for length, count in counts.items()
        if int(length) >= min_lifetime_frames and int(count) > 0
    }
    if not filtered:
        return {
            "time_ps": np.array([], dtype=float),
            "survival": np.array([], dtype=float),
            "n_events": 0,
            "mean_lifetime_ps": np.nan,
            "median_lifetime_ps": np.nan,
            "max_lifetime_ps": np.nan,
        }

    max_len = max(filtered)
    count_array = np.zeros(max_len + 1, dtype=np.int64)
    for length, count in filtered.items():
        count_array[length] += count

    total = int(count_array.sum())
    tail_counts = np.cumsum(count_array[::-1])[::-1]
    survival = tail_counts[1:] / total
    time_ps = np.arange(max_len, dtype=float) * dt_ps

    lengths = np.arange(max_len + 1)
    mean_frames = float(np.sum(lengths * count_array) / total)
    cumulative = np.cumsum(count_array)
    median_frame = int(np.searchsorted(cumulative, (total + 1) / 2.0))

    return {
        "time_ps": time_ps,
        "survival": survival.astype(float),
        "n_events": total,
        "mean_lifetime_ps": mean_frames * dt_ps,
        "median_lifetime_ps": median_frame * dt_ps,
        "max_lifetime_ps": max_len * dt_ps,
    }


def get_valid_box(ts, use_pbc):
    if not use_pbc or ts.dimensions is None:
        return None
    if np.any(~np.isfinite(ts.dimensions[:3])) or np.any(ts.dimensions[:3] <= 0.0):
        return None
    return ts.dimensions


def get_frame_iterator(universe, start_frame, end_frame, skip_frame):
    slice_end = None if end_frame == -1 else end_frame
    return universe.trajectory[start_frame:slice_end:skip_frame]


def collect_bb_indices(residues, atom_name):
    indices = []
    for residue in residues:
        bb = residue.atoms.select_atoms(f"name {atom_name}")
        if len(bb) != 1:
            raise ValueError(
                f"Expected exactly one {atom_name} atom in residue "
                f"{residue.resid} {residue.resname}, got {len(bb)}"
            )
        indices.append(bb[0].index)
    return indices


def build_chain_bb_indices(
    universe,
    residues_per_chain=RESIDUES_PER_CHAIN,
    nac_residue_range=NAC_RESIDUE_RANGE,
    atom_name=NAC_BACKBONE_NAME,
):
    protein_residues = universe.select_atoms("protein").residues
    n_total_res = len(protein_residues)

    if n_total_res % residues_per_chain != 0:
        raise ValueError(
            f"Total protein residues {n_total_res} is not divisible by "
            f"RESIDUES_PER_CHAIN={residues_per_chain}."
        )

    nac_start, nac_end = nac_residue_range
    if nac_start < 1 or nac_end > residues_per_chain or nac_end < nac_start:
        raise ValueError(
            f"Invalid NAC residue range {nac_residue_range} for "
            f"RESIDUES_PER_CHAIN={residues_per_chain}."
        )

    n_chains = n_total_res // residues_per_chain
    chain_bb_indices = []
    nac_bb_indices = []

    for chain_idx in range(n_chains):
        offset = chain_idx * residues_per_chain
        chain_residues = protein_residues[offset : offset + residues_per_chain]
        nac_residues = chain_residues[nac_start - 1 : nac_end]

        chain_bb_indices.append(collect_bb_indices(chain_residues, atom_name))
        nac_bb_indices.append(collect_bb_indices(nac_residues, atom_name))

    return (
        np.asarray(chain_bb_indices, dtype=np.int64),
        np.asarray(nac_bb_indices, dtype=np.int64),
        n_chains,
    )


def build_chain_pairs(n_chains):
    pairs = [(i, j) for i in range(n_chains - 1) for j in range(i + 1, n_chains)]
    if not pairs:
        raise ValueError("At least two protein chains are required.")
    return np.asarray(pairs, dtype=np.int32)


def build_pair_index_matrix(n_chains, pairs):
    pair_index = np.full((n_chains, n_chains), -1, dtype=np.int32)
    for idx, (chain_i, chain_j) in enumerate(pairs):
        pair_index[chain_i, chain_j] = idx
        pair_index[chain_j, chain_i] = idx
    return pair_index


def build_flat_chain_residue_ids(n_chains, n_residues):
    chain_ids = np.repeat(np.arange(n_chains, dtype=np.int32), n_residues)
    residue_idx = np.tile(np.arange(n_residues, dtype=np.int16), n_chains)
    return chain_ids, residue_idx


def corresponding_residue_contacts_from_close_pairs(
    close_pairs,
    flat_chain_ids,
    flat_residue_idx,
    pair_index,
    full_to_valid_pair,
    valid_chain_mask,
    n_valid_pairs,
    n_residues,
):
    residue_contacts = np.zeros((n_valid_pairs, n_residues), dtype=bool)
    if len(close_pairs) == 0:
        return residue_contacts

    atom_a = close_pairs[:, 0]
    atom_b = close_pairs[:, 1]
    chain_a = flat_chain_ids[atom_a]
    chain_b = flat_chain_ids[atom_b]
    keep = (
        (chain_a != chain_b)
        & valid_chain_mask[chain_a]
        & valid_chain_mask[chain_b]
    )
    if not np.any(keep):
        return residue_contacts

    atom_a = atom_a[keep]
    atom_b = atom_b[keep]
    chain_a = chain_a[keep]
    chain_b = chain_b[keep]
    residue_a = flat_residue_idx[atom_a]
    residue_b = flat_residue_idx[atom_b]

    a_is_source = chain_a < chain_b
    source_chain = np.where(a_is_source, chain_a, chain_b)
    partner_chain = np.where(a_is_source, chain_b, chain_a)
    source_residue = np.where(a_is_source, residue_a, residue_b)
    partner_residue = np.where(a_is_source, residue_b, residue_a)

    corresponding_keep = source_residue == partner_residue
    if not np.any(corresponding_keep):
        return residue_contacts

    source_residue = source_residue[corresponding_keep]
    full_pair_idx = pair_index[
        source_chain[corresponding_keep],
        partner_chain[corresponding_keep],
    ]
    pair_valid = full_pair_idx >= 0
    if not np.any(pair_valid):
        return residue_contacts

    valid_pair_idx = full_to_valid_pair[full_pair_idx[pair_valid]]
    valid = valid_pair_idx >= 0
    if np.any(valid):
        residue_contacts[valid_pair_idx[valid], source_residue[pair_valid][valid]] = True

    return residue_contacts


def build_corresponding_contact_masks(
    residue_contacts,
    contact_modes,
    n_triplets,
    triplet_size=NAC_TRIPLET_SIZE,
):
    masks = {}
    if "corresponding_nac_bb" in contact_modes:
        masks["corresponding_nac_bb"] = np.any(residue_contacts, axis=1)
    if "three_corresponding_nac_bb" in contact_modes:
        masks["three_corresponding_nac_bb"] = np.sum(residue_contacts, axis=1) >= 3
    if "triplet_corresponding_nac_bb" in contact_modes:
        triplet_contacts = np.ones((residue_contacts.shape[0], n_triplets), dtype=bool)
        for offset in range(triplet_size):
            triplet_contacts &= residue_contacts[:, offset : offset + n_triplets]
        masks["triplet_corresponding_nac_bb"] = np.any(triplet_contacts, axis=1)
    return masks


def build_chain_selection_mask(n_chains, chain_selection=None):
    selected = np.ones(n_chains, dtype=bool)

    if chain_selection is None or chain_selection == "all":
        label = "all"
    elif chain_selection == "last_40":
        selected[:] = False
        selected[max(0, n_chains - 40) :] = True
        label = "last_40"
    elif isinstance(chain_selection, int):
        selected[:] = False
        selected[max(0, n_chains - chain_selection) :] = True
        label = f"last_{chain_selection}"
    elif isinstance(chain_selection, slice):
        selected[:] = False
        selected[np.arange(n_chains)[chain_selection]] = True
        label = f"slice_{chain_selection.start}_{chain_selection.stop}_{chain_selection.step}"
    else:
        raise ValueError(f"Unsupported chain_selection={chain_selection!r}")

    selected_indices = np.where(selected)[0]
    return selected, label, selected_indices


def estimate_valid_chains(
    universe,
    chain_bb_indices,
    frame_iterator,
    radius_limit_A,
    progress_every=100,
):
    n_frames_actual = len(frame_iterator)
    n_chains = chain_bb_indices.shape[0]
    max_dist_origin = np.zeros(n_chains, dtype=float)
    masses = universe.atoms.masses[chain_bb_indices]
    mass_sums = masses.sum(axis=1)
    use_mass_weighted_centers = np.all(np.isfinite(mass_sums)) and np.all(mass_sums > 0.0)

    for frame_idx, _ in enumerate(frame_iterator):
        if frame_idx % progress_every == 0:
            print(f"     radius frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        positions = universe.atoms.positions[chain_bb_indices]
        if use_mass_weighted_centers:
            centers = np.einsum("cn,cnx->cx", masses, positions) / mass_sums[:, None]
        else:
            centers = positions.mean(axis=1)
        dist_origin = np.linalg.norm(centers, axis=1)
        max_dist_origin = np.maximum(max_dist_origin, dist_origin)

    print(f"     radius frame {n_frames_actual:>5}/{n_frames_actual}")
    return max_dist_origin <= radius_limit_A


def estimate_persistent_largest_cluster_chains(
    universe,
    chain_bb_indices,
    frame_iterator,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    use_pbc=NAC_USE_PBC,
    progress_every=100,
):
    n_frames_actual = len(frame_iterator)
    n_chains, n_bb_per_chain = chain_bb_indices.shape
    bb_indices = chain_bb_indices.reshape(-1)
    bb_to_chain = np.repeat(np.arange(n_chains, dtype=np.int32), n_bb_per_chain)
    valid_mask = np.ones(n_chains, dtype=bool)

    for frame_idx, ts in enumerate(frame_iterator):
        if frame_idx % progress_every == 0:
            print(f"     cluster frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        box = get_valid_box(ts, use_pbc)
        pairs = capped_distance(
            universe.atoms.positions[bb_indices],
            universe.atoms.positions[bb_indices],
            max_cutoff=cluster_dist_cutoff_A,
            box=box,
            return_distances=False,
        )
        chain_pairs = bb_to_chain[pairs]
        chain_pairs = chain_pairs[chain_pairs[:, 0] != chain_pairs[:, 1]]

        if len(chain_pairs) > 0:
            rows = np.concatenate([chain_pairs[:, 0], chain_pairs[:, 1]])
            cols = np.concatenate([chain_pairs[:, 1], chain_pairs[:, 0]])
            data = np.ones(len(rows), dtype=np.int8)
            adjacency = csr_matrix((data, (rows, cols)), shape=(n_chains, n_chains))
        else:
            adjacency = csr_matrix((n_chains, n_chains))

        n_components, labels = connected_components(adjacency, directed=False)
        if n_components > 0:
            largest_label = np.argmax(np.bincount(labels))
            valid_mask &= labels == largest_label
        else:
            valid_mask[:] = False
            break

    print(f"     cluster frame {n_frames_actual:>5}/{n_frames_actual}")
    return valid_mask


def resolve_chain_filter_method(use_radius_filter=True, chain_filter_method=None):
    if chain_filter_method is None:
        return "radius" if use_radius_filter else "none"

    choices = {"none", "radius", "largest_cluster"}
    if chain_filter_method not in choices:
        raise ValueError(
            f"Unknown chain_filter_method={chain_filter_method!r}. "
            f"Choices: {', '.join(sorted(choices))}"
        )
    return chain_filter_method


def select_valid_chains(
    universe,
    chain_bb_indices,
    start_frame,
    end_frame,
    skip_frame,
    selected_chain_mask,
    radius_limit_A,
    use_radius_filter=True,
    chain_filter_method=None,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    use_pbc=NAC_USE_PBC,
    progress_every=100,
):
    method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )

    if method == "largest_cluster":
        valid_chain_mask = estimate_persistent_largest_cluster_chains(
            universe,
            chain_bb_indices,
            get_frame_iterator(universe, start_frame, end_frame, skip_frame),
            cluster_dist_cutoff_A=cluster_dist_cutoff_A,
            use_pbc=use_pbc,
            progress_every=progress_every,
        )
    elif method == "radius":
        valid_chain_mask = estimate_valid_chains(
            universe,
            chain_bb_indices,
            get_frame_iterator(universe, start_frame, end_frame, skip_frame),
            radius_limit_A,
            progress_every=progress_every,
        )
    else:
        valid_chain_mask = np.ones(chain_bb_indices.shape[0], dtype=bool)

    valid_chain_mask &= selected_chain_mask
    return valid_chain_mask, method


def file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def valid_chain_cache_path(
    cache_dir,
    tpr_file,
    xtc_file,
    start_frame,
    end_frame,
    skip_frame,
    selected_chain_mask,
    radius_limit_A,
    use_radius_filter,
    chain_filter_method,
    cluster_dist_cutoff_A,
    use_pbc,
    residues_per_chain,
    nac_residue_range,
    backbone_name,
):
    method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )
    payload = {
        "tpr": file_fingerprint(tpr_file),
        "xtc": file_fingerprint(xtc_file),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "skip_frame": int(skip_frame),
        "selected_chain_indices": np.where(selected_chain_mask)[0].astype(int).tolist(),
        "radius_limit_A": float(radius_limit_A),
        "chain_filter_method": method,
        "cluster_dist_cutoff_A": float(cluster_dist_cutoff_A),
        "use_pbc": bool(use_pbc),
        "residues_per_chain": int(residues_per_chain),
        "nac_residue_range": [int(nac_residue_range[0]), int(nac_residue_range[1])],
        "backbone_name": str(backbone_name),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, f"valid_chains_{digest}.npz")


def select_valid_chains_cached(
    universe,
    chain_bb_indices,
    start_frame,
    end_frame,
    skip_frame,
    selected_chain_mask,
    radius_limit_A,
    tpr_file,
    xtc_file,
    residues_per_chain=RESIDUES_PER_CHAIN,
    nac_residue_range=NAC_RESIDUE_RANGE,
    backbone_name=NAC_BACKBONE_NAME,
    use_radius_filter=True,
    chain_filter_method=None,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    use_pbc=NAC_USE_PBC,
    progress_every=100,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )
    cache_dir = valid_chain_cache_dir or DEFAULT_VALID_CHAIN_CACHE_DIR
    cache_path = None

    if use_valid_chain_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = valid_chain_cache_path(
            cache_dir,
            tpr_file,
            xtc_file,
            start_frame,
            end_frame,
            skip_frame,
            selected_chain_mask,
            radius_limit_A,
            use_radius_filter,
            method,
            cluster_dist_cutoff_A,
            use_pbc,
            residues_per_chain,
            nac_residue_range,
            backbone_name,
        )
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            valid_chain_mask = cached["valid_chain_mask"].astype(bool)
            if valid_chain_mask.shape == (chain_bb_indices.shape[0],):
                print(f"     loaded valid-chain cache: {cache_path}")
                return valid_chain_mask, method
            print(f"     ignoring incompatible valid-chain cache: {cache_path}")

    valid_chain_mask, method = select_valid_chains(
        universe,
        chain_bb_indices,
        start_frame,
        end_frame,
        skip_frame,
        selected_chain_mask,
        radius_limit_A,
        use_radius_filter=use_radius_filter,
        chain_filter_method=method,
        cluster_dist_cutoff_A=cluster_dist_cutoff_A,
        use_pbc=use_pbc,
        progress_every=progress_every,
    )

    if use_valid_chain_cache and cache_path is not None:
        np.savez_compressed(
            cache_path,
            valid_chain_mask=valid_chain_mask.astype(np.bool_),
            chain_filter_method=np.array(method),
        )
        print(f"     saved valid-chain cache: {cache_path}")

    return valid_chain_mask, method


def cache_matches_chain_filter(
    summary_df,
    chain_filter_method,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    radius_limit_A=100.0,
):
    if summary_df.empty:
        return True
    if "chain_filter_method" not in summary_df:
        return False

    methods = set(summary_df["chain_filter_method"].dropna().astype(str))
    if methods != {chain_filter_method}:
        return False

    if chain_filter_method == "largest_cluster":
        if "cluster_dist_cutoff_A" not in summary_df:
            return False
        return np.allclose(
            summary_df["cluster_dist_cutoff_A"].to_numpy(dtype=float),
            float(cluster_dist_cutoff_A),
            equal_nan=False,
        )

    if chain_filter_method == "radius":
        if "radius_limit_A" not in summary_df:
            return False
        return np.allclose(
            summary_df["radius_limit_A"].to_numpy(dtype=float),
            float(radius_limit_A),
            equal_nan=False,
        )

    return True


def analyze_single_nac_contact_trajectory(
    tpr_file,
    xtc_file,
    start_frame,
    end_frame,
    skip_frame,
    radius_limit_A,
    nac_residue_range=NAC_RESIDUE_RANGE,
    residues_per_chain=RESIDUES_PER_CHAIN,
    triplet_size=NAC_TRIPLET_SIZE,
    backbone_name=NAC_BACKBONE_NAME,
    contact_cutoff_nm=NAC_CONTACT_CUTOFF_NM,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    min_lifetime_frames=NAC_MIN_LIFETIME_FRAMES,
    contact_mode="corresponding_nac_bb",
    chain_selection=None,
    progress_every=100,
):
    mode_cfg = get_contact_mode_config(contact_mode)
    print(f"  -> {mode_cfg['label']}: {os.path.basename(xtc_file)}")
    universe = mda.Universe(tpr_file, xtc_file)
    dt_eff = universe.trajectory.dt * skip_frame
    contact_cutoff_A = contact_cutoff_nm * 10.0

    chain_bb_indices, nac_bb_indices, n_chains = build_chain_bb_indices(
        universe,
        residues_per_chain=residues_per_chain,
        nac_residue_range=nac_residue_range,
        atom_name=backbone_name,
    )
    n_nac = nac_bb_indices.shape[1]
    n_triplets = n_nac - triplet_size + 1
    if contact_mode == "triplet_corresponding_nac_bb" and n_triplets <= 0:
        raise ValueError(f"NAC length {n_nac} is shorter than triplet_size={triplet_size}.")

    selected_chain_mask, chain_selection_label, selected_chain_indices = build_chain_selection_mask(
        n_chains,
        chain_selection=chain_selection,
    )
    n_selected_chains = int(np.sum(selected_chain_mask))
    if n_selected_chains < 2:
        raise RuntimeError(
            f"Fewer than two selected chains in {xtc_file}: "
            f"chain_selection={chain_selection_label}, n_selected={n_selected_chains}"
        )

    pairs = build_chain_pairs(n_chains)
    pair_index = build_pair_index_matrix(n_chains, pairs)
    n_pairs = len(pairs)

    frame_iterator = get_frame_iterator(universe, start_frame, end_frame, skip_frame)
    n_frames_actual = len(frame_iterator)
    if n_frames_actual < 2:
        raise RuntimeError(f"Too few frames in {xtc_file}: {n_frames_actual}")

    valid_chain_mask, resolved_chain_filter_method = select_valid_chains(
        universe,
        chain_bb_indices,
        start_frame,
        end_frame,
        skip_frame,
        selected_chain_mask,
        radius_limit_A,
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
        cluster_dist_cutoff_A=cluster_dist_cutoff_A,
        use_pbc=use_pbc,
        progress_every=progress_every,
    )
    n_valid_chains = int(np.sum(valid_chain_mask))
    if n_valid_chains < 2:
        raise RuntimeError(f"Fewer than two valid chains in {xtc_file}")

    valid_pair_mask_all = valid_chain_mask[pairs[:, 0]] & valid_chain_mask[pairs[:, 1]]
    n_valid_pairs = int(np.sum(valid_pair_mask_all))

    if contact_mode in {
        "any_nac_to_nac_bb",
        "three_corresponding_nac_bb",
        "corresponding_nac_bb",
        "triplet_corresponding_nac_bb",
    }:
        n_contact_sites = 1
        active_lengths = np.zeros(n_pairs, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported contact_mode={contact_mode!r}")

    if contact_mode == "any_nac_to_nac_bb":
        nac_flat_chain_ids = np.repeat(np.arange(n_chains, dtype=np.int32), n_nac)

    lifetime_counts = defaultdict(int)

    for frame_idx, ts in enumerate(get_frame_iterator(universe, start_frame, end_frame, skip_frame)):
        if frame_idx % progress_every == 0:
            print(f"     contact frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        nac_positions = universe.atoms.positions[nac_bb_indices]
        box = get_valid_box(ts, use_pbc)

        if contact_mode == "any_nac_to_nac_bb":
            contact_mask = np.zeros(n_pairs, dtype=bool)
            close_pairs = capped_distance(
                nac_positions.reshape(-1, 3),
                nac_positions.reshape(-1, 3),
                max_cutoff=contact_cutoff_A,
                box=box,
                return_distances=False,
            )

            if len(close_pairs) > 0:
                nac_chain = nac_flat_chain_ids[close_pairs[:, 0]]
                other_nac_chain = nac_flat_chain_ids[close_pairs[:, 1]]
                keep = (
                    (nac_chain != other_nac_chain)
                    & valid_chain_mask[nac_chain]
                    & valid_chain_mask[other_nac_chain]
                )
                if np.any(keep):
                    contact_pair_idx = pair_index[nac_chain[keep], other_nac_chain[keep]]
                    contact_pair_idx = contact_pair_idx[contact_pair_idx >= 0]
                    contact_mask[contact_pair_idx] = True

            update_active_lifetimes(active_lengths, contact_mask, lifetime_counts)
        else:
            for pair_start in range(0, n_pairs, pair_chunk):
                pair_end = min(pair_start + pair_chunk, n_pairs)
                pair_block = pairs[pair_start:pair_end]
                chain_i = pair_block[:, 0]
                chain_j = pair_block[:, 1]
                valid_pair_mask = valid_chain_mask[chain_i] & valid_chain_mask[chain_j]

                pos_i = nac_positions[chain_i].reshape(-1, 3)
                pos_j = nac_positions[chain_j].reshape(-1, 3)
                distances = calc_bonds(pos_i, pos_j, box=box).reshape(len(pair_block), n_nac)
                residue_contacts = distances <= contact_cutoff_A

                if contact_mode == "triplet_corresponding_nac_bb":
                    triplet_contacts = np.ones((len(pair_block), n_triplets), dtype=bool)
                    for offset in range(triplet_size):
                        triplet_contacts &= residue_contacts[:, offset : offset + n_triplets]
                    contact_mask = np.any(triplet_contacts, axis=1)
                elif contact_mode == "three_corresponding_nac_bb":
                    contact_mask = np.sum(residue_contacts, axis=1) >= 3
                else:
                    contact_mask = np.any(residue_contacts, axis=1)

                contact_mask &= valid_pair_mask
                update_active_lifetimes(
                    active_lengths[pair_start:pair_end],
                    contact_mask,
                    lifetime_counts,
                )

    print(f"     contact frame {n_frames_actual:>5}/{n_frames_actual}")
    add_lengths_to_counts(lifetime_counts, active_lengths[active_lengths > 0])

    survival = counts_to_survival(
        lifetime_counts,
        dt_eff,
        min_lifetime_frames=min_lifetime_frames,
    )
    survival.update(
        {
            "dt_ps": float(dt_eff),
            "n_frames": int(n_frames_actual),
            "n_chains": int(n_chains),
            "chain_selection": chain_selection_label,
            "n_selected_chains": int(n_selected_chains),
            "selected_chain_start": int(selected_chain_indices[0]) if len(selected_chain_indices) > 0 else -1,
            "selected_chain_end": int(selected_chain_indices[-1]) if len(selected_chain_indices) > 0 else -1,
            "chain_filter_method": resolved_chain_filter_method,
            "cluster_dist_cutoff_A": float(cluster_dist_cutoff_A),
            "radius_limit_A": float(radius_limit_A),
            "n_valid_chains": int(n_valid_chains),
            "n_pairs": int(n_pairs),
            "n_valid_pairs": int(n_valid_pairs),
            "n_contact_sites_per_pair": int(n_contact_sites),
            "n_triplets_per_pair": int(n_triplets),
            "contact_mode": contact_mode,
            "contact_label": mode_cfg["label"],
            "event_unit": mode_cfg["event_unit"],
            "contact_cutoff_nm": float(contact_cutoff_nm),
        }
    )
    return survival


def analyze_single_nac_triplet_trajectory(*args, **kwargs):
    kwargs["contact_mode"] = "triplet_corresponding_nac_bb"
    return analyze_single_nac_contact_trajectory(*args, **kwargs)


def analyze_single_nac_contact_trajectory_multi(
    tpr_file,
    xtc_file,
    start_frame,
    end_frame,
    skip_frame,
    radius_limit_A,
    contact_modes=DEFAULT_CONTACT_MODES,
    nac_residue_range=NAC_RESIDUE_RANGE,
    residues_per_chain=RESIDUES_PER_CHAIN,
    triplet_size=NAC_TRIPLET_SIZE,
    backbone_name=NAC_BACKBONE_NAME,
    contact_cutoff_nm=NAC_CONTACT_CUTOFF_NM,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    min_lifetime_frames=NAC_MIN_LIFETIME_FRAMES,
    chain_selection=None,
    progress_every=100,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    contact_modes = tuple(dict.fromkeys(contact_modes))
    for contact_mode in contact_modes:
        get_contact_mode_config(contact_mode)

    print(
        "  -> shared survival scan for "
        f"{len(contact_modes)} contact modes: {os.path.basename(xtc_file)}"
    )
    universe = mda.Universe(tpr_file, xtc_file)
    dt_eff = universe.trajectory.dt * skip_frame
    contact_cutoff_A = contact_cutoff_nm * 10.0

    chain_bb_indices, nac_bb_indices, n_chains = build_chain_bb_indices(
        universe,
        residues_per_chain=residues_per_chain,
        nac_residue_range=nac_residue_range,
        atom_name=backbone_name,
    )
    n_nac = nac_bb_indices.shape[1]
    n_triplets = n_nac - triplet_size + 1
    if "triplet_corresponding_nac_bb" in contact_modes and n_triplets <= 0:
        raise ValueError(f"NAC length {n_nac} is shorter than triplet_size={triplet_size}.")

    selected_chain_mask, chain_selection_label, selected_chain_indices = build_chain_selection_mask(
        n_chains,
        chain_selection=chain_selection,
    )
    n_selected_chains = int(np.sum(selected_chain_mask))
    if n_selected_chains < 2:
        raise RuntimeError(
            f"Fewer than two selected chains in {xtc_file}: "
            f"chain_selection={chain_selection_label}, n_selected={n_selected_chains}"
        )

    pairs = build_chain_pairs(n_chains)
    pair_index = build_pair_index_matrix(n_chains, pairs)
    n_pairs = len(pairs)

    frame_iterator = get_frame_iterator(universe, start_frame, end_frame, skip_frame)
    n_frames_actual = len(frame_iterator)
    if n_frames_actual < 2:
        raise RuntimeError(f"Too few frames in {xtc_file}: {n_frames_actual}")

    valid_chain_mask, resolved_chain_filter_method = select_valid_chains_cached(
        universe,
        chain_bb_indices,
        start_frame,
        end_frame,
        skip_frame,
        selected_chain_mask,
        radius_limit_A,
        tpr_file,
        xtc_file,
        residues_per_chain=residues_per_chain,
        nac_residue_range=nac_residue_range,
        backbone_name=backbone_name,
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
        cluster_dist_cutoff_A=cluster_dist_cutoff_A,
        use_pbc=use_pbc,
        progress_every=progress_every,
        use_valid_chain_cache=use_valid_chain_cache,
        valid_chain_cache_dir=valid_chain_cache_dir,
    )
    n_valid_chains = int(np.sum(valid_chain_mask))
    if n_valid_chains < 2:
        raise RuntimeError(f"Fewer than two valid chains in {xtc_file}")

    valid_pair_mask_all = valid_chain_mask[pairs[:, 0]] & valid_chain_mask[pairs[:, 1]]
    valid_pair_indices = np.where(valid_pair_mask_all)[0]
    n_valid_pairs = len(valid_pair_indices)
    if n_valid_pairs == 0:
        raise RuntimeError(f"No valid chain pairs in {xtc_file}")

    full_to_valid_pair = np.full(n_pairs, -1, dtype=np.int32)
    full_to_valid_pair[valid_pair_indices] = np.arange(n_valid_pairs, dtype=np.int32)
    nac_flat_chain_ids, nac_flat_residue_idx = build_flat_chain_residue_ids(n_chains, n_nac)

    needs_any = "any_nac_to_nac_bb" in contact_modes
    corresponding_modes = tuple(mode for mode in contact_modes if mode != "any_nac_to_nac_bb")
    active_lengths = {
        contact_mode: np.zeros(n_valid_pairs, dtype=np.int32)
        for contact_mode in contact_modes
    }
    lifetime_counts = {contact_mode: defaultdict(int) for contact_mode in contact_modes}

    for frame_idx, ts in enumerate(get_frame_iterator(universe, start_frame, end_frame, skip_frame)):
        if frame_idx % progress_every == 0:
            print(f"     contact frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        nac_positions = universe.atoms.positions[nac_bb_indices]
        box = get_valid_box(ts, use_pbc)

        if needs_any:
            close_pairs = capped_distance(
                nac_positions.reshape(-1, 3),
                nac_positions.reshape(-1, 3),
                max_cutoff=contact_cutoff_A,
                box=box,
                return_distances=False,
            )
            contact_mask = np.zeros(n_pairs, dtype=bool)

            if len(close_pairs) > 0:
                nac_chain = nac_flat_chain_ids[close_pairs[:, 0]]
                other_nac_chain = nac_flat_chain_ids[close_pairs[:, 1]]
                keep = (
                    (nac_chain != other_nac_chain)
                    & valid_chain_mask[nac_chain]
                    & valid_chain_mask[other_nac_chain]
                )
                if np.any(keep):
                    contact_pair_idx = pair_index[nac_chain[keep], other_nac_chain[keep]]
                    contact_pair_idx = contact_pair_idx[contact_pair_idx >= 0]
                    contact_mask[contact_pair_idx] = True

            update_active_lifetimes(
                active_lengths["any_nac_to_nac_bb"],
                contact_mask[valid_pair_indices],
                lifetime_counts["any_nac_to_nac_bb"],
            )

            if corresponding_modes:
                residue_contacts = corresponding_residue_contacts_from_close_pairs(
                    close_pairs,
                    nac_flat_chain_ids,
                    nac_flat_residue_idx,
                    pair_index,
                    full_to_valid_pair,
                    valid_chain_mask,
                    n_valid_pairs,
                    n_nac,
                )
                masks = build_corresponding_contact_masks(
                    residue_contacts,
                    corresponding_modes,
                    n_triplets,
                    triplet_size=triplet_size,
                )
                for contact_mode, contact_mask in masks.items():
                    update_active_lifetimes(
                        active_lengths[contact_mode],
                        contact_mask,
                        lifetime_counts[contact_mode],
                    )
        elif corresponding_modes:
            for pair_start in range(0, n_valid_pairs, pair_chunk):
                pair_end = min(pair_start + pair_chunk, n_valid_pairs)
                pair_block = pairs[valid_pair_indices[pair_start:pair_end]]
                chain_i = pair_block[:, 0]
                chain_j = pair_block[:, 1]

                pos_i = nac_positions[chain_i].reshape(-1, 3)
                pos_j = nac_positions[chain_j].reshape(-1, 3)
                distances = calc_bonds(pos_i, pos_j, box=box).reshape(len(pair_block), n_nac)
                residue_contacts = distances <= contact_cutoff_A
                masks = build_corresponding_contact_masks(
                    residue_contacts,
                    corresponding_modes,
                    n_triplets,
                    triplet_size=triplet_size,
                )
                for contact_mode, contact_mask in masks.items():
                    update_active_lifetimes(
                        active_lengths[contact_mode][pair_start:pair_end],
                        contact_mask,
                        lifetime_counts[contact_mode],
                    )

    print(f"     contact frame {n_frames_actual:>5}/{n_frames_actual}")

    results = {}
    for contact_mode in contact_modes:
        mode_cfg = get_contact_mode_config(contact_mode)
        add_lengths_to_counts(
            lifetime_counts[contact_mode],
            active_lengths[contact_mode][active_lengths[contact_mode] > 0],
        )
        survival = counts_to_survival(
            lifetime_counts[contact_mode],
            dt_eff,
            min_lifetime_frames=min_lifetime_frames,
        )
        survival.update(
            {
                "dt_ps": float(dt_eff),
                "n_frames": int(n_frames_actual),
                "n_chains": int(n_chains),
                "chain_selection": chain_selection_label,
                "n_selected_chains": int(n_selected_chains),
                "selected_chain_start": int(selected_chain_indices[0]) if len(selected_chain_indices) > 0 else -1,
                "selected_chain_end": int(selected_chain_indices[-1]) if len(selected_chain_indices) > 0 else -1,
                "chain_filter_method": resolved_chain_filter_method,
                "cluster_dist_cutoff_A": float(cluster_dist_cutoff_A),
                "radius_limit_A": float(radius_limit_A),
                "n_valid_chains": int(n_valid_chains),
                "n_pairs": int(n_pairs),
                "n_valid_pairs": int(n_valid_pairs),
                "n_contact_sites_per_pair": 1,
                "n_triplets_per_pair": int(n_triplets),
                "contact_mode": contact_mode,
                "contact_label": mode_cfg["label"],
                "event_unit": mode_cfg["event_unit"],
                "contact_cutoff_nm": float(contact_cutoff_nm),
            }
        )
        results[contact_mode] = survival

    return results


def build_survival_stack(time_list, survival_list, dt_hint):
    t_max = max(t[-1] for t in time_list)
    dt_common = max(
        np.median(np.diff(t)) if len(t) > 1 else dt_hint
        for t in time_list
    )
    npts = int(np.floor(t_max / dt_common)) + 1
    t_common = np.arange(npts, dtype=float) * dt_common
    survival_stack = np.array(
        [
            np.interp(t_common, t, survival, left=1.0, right=0.0)
            for t, survival in zip(time_list, survival_list)
        ]
    )
    return t_common, survival_stack


def resolve_survival_output_paths(output_dir, contact_mode):
    mode_cfg = get_contact_mode_config(contact_mode)
    mode_output_dir = os.path.join(output_dir, mode_cfg["file_prefix"])
    return {
        "mode_output_dir": mode_output_dir,
        "cache_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_survival_data.csv"),
        "summary_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_relaxation_summary.csv"),
        "trajectory_summary_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_trajectory_summary.csv"),
    }


def run_nac_contact_relaxation_analysis_multi(
    contact_modes=DEFAULT_CONTACT_MODES,
    groups=None,
    group_labels=None,
    group_colors=None,
    radius_limit_A=100.0,
    start_key="start",
    end_frame=-1,
    skip_frame=1,
    progress_every=100,
    output_dir=None,
    nac_residue_range=NAC_RESIDUE_RANGE,
    residues_per_chain=RESIDUES_PER_CHAIN,
    triplet_size=NAC_TRIPLET_SIZE,
    backbone_name=NAC_BACKBONE_NAME,
    contact_cutoff_nm=NAC_CONTACT_CUTOFF_NM,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    min_lifetime_frames=NAC_MIN_LIFETIME_FRAMES,
    force=False,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    groups = groups or DEFAULT_GROUPS
    contact_modes = tuple(dict.fromkeys(contact_modes))
    group_labels = group_labels or {group["name"]: group.get("label", group["name"]) for group in groups}
    group_colors = group_colors or {group["name"]: group.get("color", "#1f77b4") for group in groups}
    resolved_chain_filter_method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    modes_to_run = []
    for contact_mode in contact_modes:
        paths = resolve_survival_output_paths(output_dir, contact_mode)
        if not force and os.path.exists(paths["cache_csv"]) and os.path.exists(paths["summary_csv"]):
            print(f"Detected cached {contact_mode} survival data; loading.")
            df_all = pd.read_csv(paths["cache_csv"])
            summary_df = pd.read_csv(paths["summary_csv"])
            if cache_matches_chain_filter(
                summary_df,
                resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                radius_limit_A=radius_limit_A,
            ):
                results[contact_mode] = {"data": df_all, "summary": summary_df}
                continue
            print(f"Cached {contact_mode} survival data use different chain filtering; recalculating.")
        modes_to_run.append(contact_mode)

    if not modes_to_run:
        combined_summary = pd.concat(
            [results[mode]["summary"] for mode in contact_modes if mode in results],
            ignore_index=True,
        )
        return results, combined_summary

    rows_by_mode = {mode: [] for mode in modes_to_run}
    summary_rows_by_mode = {mode: [] for mode in modes_to_run}
    traj_summary_rows_by_mode = {mode: [] for mode in modes_to_run}

    for group in groups:
        group_name = group["name"]
        group_label = group_labels.get(group_name, group.get("label", group_name))

        print("=" * 80)
        print(f"Processing shared corresponding survival scan: {group_label}")
        group_chain_selection = group.get(
            "chain_selection",
            DEFAULT_GROUP_CHAIN_SELECTIONS.get(group_name, "all"),
        )
        print(f"Chain selection: {group_chain_selection}")

        time_lists = {mode: [] for mode in modes_to_run}
        survival_lists = {mode: [] for mode in modes_to_run}
        dt_hints = {mode: [] for mode in modes_to_run}
        traj_mean_lifetimes = {mode: [] for mode in modes_to_run}
        traj_median_lifetimes = {mode: [] for mode in modes_to_run}
        traj_event_counts = {mode: [] for mode in modes_to_run}

        for traj_idx, traj in enumerate(group["trajectories"], start=1):
            traj_results = analyze_single_nac_contact_trajectory_multi(
                traj["tpr"],
                traj["xtc"],
                traj[start_key],
                end_frame,
                skip_frame,
                radius_limit_A,
                contact_modes=modes_to_run,
                nac_residue_range=nac_residue_range,
                residues_per_chain=residues_per_chain,
                triplet_size=triplet_size,
                backbone_name=backbone_name,
                contact_cutoff_nm=contact_cutoff_nm,
                pair_chunk=pair_chunk,
                use_pbc=use_pbc,
                use_radius_filter=use_radius_filter,
                chain_filter_method=resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                min_lifetime_frames=min_lifetime_frames,
                chain_selection=traj.get("chain_selection", group_chain_selection),
                progress_every=progress_every,
                use_valid_chain_cache=use_valid_chain_cache,
                valid_chain_cache_dir=valid_chain_cache_dir,
            )

            for contact_mode in modes_to_run:
                result = traj_results[contact_mode]
                mode_cfg = get_contact_mode_config(contact_mode)
                if result["n_events"] > 0:
                    traj_t_1e = crossing_time(result["time_ps"], result["survival"], 1.0 / np.e)
                    traj_t_01 = crossing_time(result["time_ps"], result["survival"], 0.1)
                else:
                    traj_t_1e = np.nan
                    traj_t_01 = np.nan
                traj_summary_rows_by_mode[contact_mode].append(
                    {
                        "group": group_name,
                        "group_label": group_label,
                        "trajectory": traj_idx,
                        "xtc": traj["xtc"],
                        "n_events": result["n_events"],
                        "mean_lifetime_ps": result["mean_lifetime_ps"],
                        "median_lifetime_ps": result["median_lifetime_ps"],
                        "max_lifetime_ps": result["max_lifetime_ps"],
                        "t_1e_ps": float(traj_t_1e),
                        "t_01_ps": float(traj_t_01),
                        "dt_ps": result["dt_ps"],
                        "n_frames": result["n_frames"],
                        "n_chains": result["n_chains"],
                        "chain_selection": result["chain_selection"],
                        "n_selected_chains": result["n_selected_chains"],
                        "selected_chain_start": result["selected_chain_start"],
                        "selected_chain_end": result["selected_chain_end"],
                        "chain_filter_method": result["chain_filter_method"],
                        "cluster_dist_cutoff_A": result["cluster_dist_cutoff_A"],
                        "radius_limit_A": result["radius_limit_A"],
                        "n_valid_chains": result["n_valid_chains"],
                        "n_pairs": result["n_pairs"],
                        "n_valid_pairs": result["n_valid_pairs"],
                        "n_contact_sites_per_pair": result["n_contact_sites_per_pair"],
                        "n_triplets_per_pair": result["n_triplets_per_pair"],
                        "contact_mode": result["contact_mode"],
                        "contact_label": result["contact_label"],
                        "event_unit": result["event_unit"],
                        "contact_cutoff_nm": result["contact_cutoff_nm"],
                    }
                )

                if result["n_events"] == 0:
                    print(f"     no {mode_cfg['label']} contact events found")
                    continue

                print(
                    f"     {mode_cfg['label']}: events={result['n_events']}, "
                    f"mean={result['mean_lifetime_ps'] / 1e6:.4f} us, "
                    f"median={result['median_lifetime_ps'] / 1e6:.4f} us"
                )

                time_lists[contact_mode].append(result["time_ps"])
                survival_lists[contact_mode].append(result["survival"])
                dt_hints[contact_mode].append(result["dt_ps"])
                traj_mean_lifetimes[contact_mode].append(result["mean_lifetime_ps"])
                traj_median_lifetimes[contact_mode].append(result["median_lifetime_ps"])
                traj_event_counts[contact_mode].append(result["n_events"])

        for contact_mode in modes_to_run:
            mode_cfg = get_contact_mode_config(contact_mode)
            if not time_lists[contact_mode]:
                summary_rows_by_mode[contact_mode].append(
                    {
                        "group": group_name,
                        "group_label": group_label,
                        "observable": mode_cfg["observable"],
                        "contact_mode": contact_mode,
                        "contact_label": mode_cfg["label"],
                        "event_unit": mode_cfg["event_unit"],
                        "chain_selection": group_chain_selection,
                        "chain_filter_method": resolved_chain_filter_method,
                        "n_events": 0,
                        "mean_lifetime_ps": np.nan,
                        "mean_lifetime_std_ps": np.nan,
                        "median_lifetime_ps": np.nan,
                        "t_1e_ps": np.nan,
                        "t_01_ps": np.nan,
                        "contact_cutoff_nm": contact_cutoff_nm,
                        "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                        "radius_limit_A": radius_limit_A,
                        "nac_start": nac_residue_range[0],
                        "nac_end": nac_residue_range[1],
                        "triplet_size": triplet_size,
                    }
                )
                continue

            t_common, survival_stack = build_survival_stack(
                time_lists[contact_mode],
                survival_lists[contact_mode],
                dt_hint=max(dt_hints[contact_mode]),
            )
            survival_mean = np.mean(survival_stack, axis=0)
            survival_std = np.std(survival_stack, axis=0)

            t_1e = crossing_time(t_common, survival_mean, 1.0 / np.e)
            t_01 = crossing_time(t_common, survival_mean, 0.1)
            mean_lifetime = float(np.nanmean(traj_mean_lifetimes[contact_mode]))
            mean_lifetime_std = float(np.nanstd(traj_mean_lifetimes[contact_mode]))
            median_lifetime = float(np.nanmedian(traj_median_lifetimes[contact_mode]))
            n_events = int(np.sum(traj_event_counts[contact_mode]))

            print(
                f"  [{mode_cfg['label']}] t(1/e)={t_1e:.2f} ps, t(0.1)={t_01:.2f} ps, "
                f"mean_lifetime={mean_lifetime:.2f} ps"
            )

            summary_rows_by_mode[contact_mode].append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "observable": mode_cfg["observable"],
                    "contact_mode": contact_mode,
                    "contact_label": mode_cfg["label"],
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "n_events": n_events,
                    "mean_lifetime_ps": mean_lifetime,
                    "mean_lifetime_std_ps": mean_lifetime_std,
                    "median_lifetime_ps": median_lifetime,
                    "t_1e_ps": float(t_1e),
                    "t_01_ps": float(t_01),
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                    "radius_limit_A": radius_limit_A,
                    "nac_start": nac_residue_range[0],
                    "nac_end": nac_residue_range[1],
                    "triplet_size": triplet_size,
                }
            )

            for idx in range(len(t_common)):
                rows_by_mode[contact_mode].append(
                    {
                        "group": group_name,
                        "group_label": group_label,
                        "observable": mode_cfg["observable"],
                        "contact_mode": contact_mode,
                        "contact_label": mode_cfg["label"],
                        "event_unit": mode_cfg["event_unit"],
                        "chain_selection": group_chain_selection,
                        "chain_filter_method": resolved_chain_filter_method,
                        "time_ps": float(t_common[idx]),
                        "survival_mean": float(survival_mean[idx]),
                        "survival_std": float(survival_std[idx]),
                        "t_1e_ps": float(t_1e),
                        "t_01_ps": float(t_01),
                        "mean_lifetime_ps": mean_lifetime,
                        "contact_cutoff_nm": contact_cutoff_nm,
                        "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                        "radius_limit_A": radius_limit_A,
                        "nac_start": nac_residue_range[0],
                        "nac_end": nac_residue_range[1],
                        "triplet_size": triplet_size,
                    }
                )

    for contact_mode in modes_to_run:
        paths = resolve_survival_output_paths(output_dir, contact_mode)
        os.makedirs(paths["mode_output_dir"], exist_ok=True)

        df_all = pd.DataFrame(rows_by_mode[contact_mode])
        summary_df = pd.DataFrame(summary_rows_by_mode[contact_mode])
        traj_summary_df = pd.DataFrame(traj_summary_rows_by_mode[contact_mode])

        df_all.to_csv(paths["cache_csv"], index=False)
        summary_df.to_csv(paths["summary_csv"], index=False)
        traj_summary_df.to_csv(paths["trajectory_summary_csv"], index=False)

        mode_cfg = get_contact_mode_config(contact_mode)
        print(f"Saved {mode_cfg['label']} survival cache: {paths['cache_csv']}")
        print(f"Saved {mode_cfg['label']} summary table: {paths['summary_csv']}")
        print(f"Saved {mode_cfg['label']} trajectory table: {paths['trajectory_summary_csv']}")
        results[contact_mode] = {"data": df_all, "summary": summary_df}

    combined_summary = pd.concat(
        [results[mode]["summary"] for mode in contact_modes if mode in results],
        ignore_index=True,
    )
    return results, combined_summary


def run_nac_contact_relaxation_analysis(
    groups,
    group_labels=None,
    group_colors=None,
    radius_limit_A=100.0,
    start_key="start",
    end_frame=-1,
    skip_frame=1,
    progress_every=100,
    cache_csv=None,
    summary_csv=None,
    trajectory_summary_csv=None,
    output_dir=None,
    nac_residue_range=NAC_RESIDUE_RANGE,
    residues_per_chain=RESIDUES_PER_CHAIN,
    triplet_size=NAC_TRIPLET_SIZE,
    backbone_name=NAC_BACKBONE_NAME,
    contact_cutoff_nm=NAC_CONTACT_CUTOFF_NM,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    min_lifetime_frames=NAC_MIN_LIFETIME_FRAMES,
    contact_mode="corresponding_nac_bb",
    force=False,
):
    group_labels = group_labels or {group["name"]: group.get("label", group["name"]) for group in groups}
    group_colors = group_colors or {group["name"]: group.get("color", "#1f77b4") for group in groups}
    mode_cfg = get_contact_mode_config(contact_mode)
    resolved_chain_filter_method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )

    if cache_csv is None:
        cache_csv = f"{mode_cfg['file_prefix']}_survival_data.csv"
    if summary_csv is None:
        summary_csv = f"{mode_cfg['file_prefix']}_relaxation_summary.csv"
    if trajectory_summary_csv is None:
        trajectory_summary_csv = f"{mode_cfg['file_prefix']}_trajectory_summary.csv"

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    mode_output_dir = os.path.join(output_dir, mode_cfg["file_prefix"])
    os.makedirs(mode_output_dir, exist_ok=True)

    def resolve_output_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(mode_output_dir, path)

    cache_csv = resolve_output_path(cache_csv)
    summary_csv = resolve_output_path(summary_csv)
    trajectory_summary_csv = resolve_output_path(trajectory_summary_csv)

    if not force and os.path.exists(cache_csv) and os.path.exists(summary_csv):
        print(f"Detected {cache_csv} and {summary_csv}, load cached NAC data...")
        df_all = pd.read_csv(cache_csv)
        summary_df = pd.read_csv(summary_csv)
        if cache_matches_chain_filter(
            summary_df,
            resolved_chain_filter_method,
            cluster_dist_cutoff_A=cluster_dist_cutoff_A,
            radius_limit_A=radius_limit_A,
        ):
            return df_all, summary_df
        print("Cached NAC data use different chain filtering; recalculating.")

    rows = []
    summary_rows = []
    traj_summary_rows = []

    for group in groups:
        group_name = group["name"]
        group_label = group_labels.get(group_name, group.get("label", group_name))

        print("=" * 80)
        print(f"Processing {mode_cfg['label']} survival: {group_label}")
        group_chain_selection = group.get(
            "chain_selection",
            DEFAULT_GROUP_CHAIN_SELECTIONS.get(group_name, "all"),
        )
        print(f"Chain selection: {group_chain_selection}")

        time_list = []
        survival_list = []
        dt_hints = []
        traj_mean_lifetimes = []
        traj_median_lifetimes = []
        traj_event_counts = []

        for traj_idx, traj in enumerate(group["trajectories"], start=1):
            result = analyze_single_nac_contact_trajectory(
                traj["tpr"],
                traj["xtc"],
                traj[start_key],
                end_frame,
                skip_frame,
                radius_limit_A,
                nac_residue_range=nac_residue_range,
                residues_per_chain=residues_per_chain,
                triplet_size=triplet_size,
                backbone_name=backbone_name,
                contact_cutoff_nm=contact_cutoff_nm,
                pair_chunk=pair_chunk,
                use_pbc=use_pbc,
                use_radius_filter=use_radius_filter,
                chain_filter_method=resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                min_lifetime_frames=min_lifetime_frames,
                contact_mode=contact_mode,
                chain_selection=traj.get("chain_selection", group_chain_selection),
                progress_every=progress_every,
            )

            if result["n_events"] > 0:
                traj_t_1e = crossing_time(result["time_ps"], result["survival"], 1.0 / np.e)
                traj_t_01 = crossing_time(result["time_ps"], result["survival"], 0.1)
            else:
                traj_t_1e = np.nan
                traj_t_01 = np.nan
            traj_summary_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "trajectory": traj_idx,
                    "xtc": traj["xtc"],
                    "n_events": result["n_events"],
                    "mean_lifetime_ps": result["mean_lifetime_ps"],
                    "median_lifetime_ps": result["median_lifetime_ps"],
                    "max_lifetime_ps": result["max_lifetime_ps"],
                    "t_1e_ps": float(traj_t_1e),
                    "t_01_ps": float(traj_t_01),
                    "dt_ps": result["dt_ps"],
                    "n_frames": result["n_frames"],
                    "n_chains": result["n_chains"],
                    "chain_selection": result["chain_selection"],
                    "n_selected_chains": result["n_selected_chains"],
                    "selected_chain_start": result["selected_chain_start"],
                    "selected_chain_end": result["selected_chain_end"],
                    "chain_filter_method": result["chain_filter_method"],
                    "cluster_dist_cutoff_A": result["cluster_dist_cutoff_A"],
                    "radius_limit_A": result["radius_limit_A"],
                    "n_valid_chains": result["n_valid_chains"],
                    "n_pairs": result["n_pairs"],
                    "n_valid_pairs": result["n_valid_pairs"],
                    "n_contact_sites_per_pair": result["n_contact_sites_per_pair"],
                    "n_triplets_per_pair": result["n_triplets_per_pair"],
                    "contact_mode": result["contact_mode"],
                    "contact_label": result["contact_label"],
                    "event_unit": result["event_unit"],
                    "contact_cutoff_nm": result["contact_cutoff_nm"],
                }
            )

            if result["n_events"] == 0:
                print(f"     no {mode_cfg['label']} contact events found")
                continue

            print(
                f"     events={result['n_events']}, "
                f"mean={result['mean_lifetime_ps'] / 1e6:.4f} us, "
                f"median={result['median_lifetime_ps'] / 1e6:.4f} us"
            )

            time_list.append(result["time_ps"])
            survival_list.append(result["survival"])
            dt_hints.append(result["dt_ps"])
            traj_mean_lifetimes.append(result["mean_lifetime_ps"])
            traj_median_lifetimes.append(result["median_lifetime_ps"])
            traj_event_counts.append(result["n_events"])

        if not time_list:
            summary_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "observable": mode_cfg["observable"],
                    "contact_mode": contact_mode,
                    "contact_label": mode_cfg["label"],
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "n_events": 0,
                    "mean_lifetime_ps": np.nan,
                    "mean_lifetime_std_ps": np.nan,
                    "median_lifetime_ps": np.nan,
                    "t_1e_ps": np.nan,
                    "t_01_ps": np.nan,
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                    "radius_limit_A": radius_limit_A,
                    "nac_start": nac_residue_range[0],
                    "nac_end": nac_residue_range[1],
                    "triplet_size": triplet_size,
                }
            )
            continue

        t_common, survival_stack = build_survival_stack(time_list, survival_list, dt_hint=max(dt_hints))
        survival_mean = np.mean(survival_stack, axis=0)
        survival_std = np.std(survival_stack, axis=0)

        t_1e = crossing_time(t_common, survival_mean, 1.0 / np.e)
        t_01 = crossing_time(t_common, survival_mean, 0.1)

        mean_lifetime = float(np.nanmean(traj_mean_lifetimes))
        mean_lifetime_std = float(np.nanstd(traj_mean_lifetimes))
        median_lifetime = float(np.nanmedian(traj_median_lifetimes))
        n_events = int(np.sum(traj_event_counts))

        print(
            f"  [{mode_cfg['label']}] t(1/e)={t_1e:.2f} ps, t(0.1)={t_01:.2f} ps, "
            f"mean_lifetime={mean_lifetime:.2f} ps"
        )

        summary_rows.append(
            {
                "group": group_name,
                "group_label": group_label,
                "observable": mode_cfg["observable"],
                "contact_mode": contact_mode,
                "contact_label": mode_cfg["label"],
                "event_unit": mode_cfg["event_unit"],
                "chain_selection": group_chain_selection,
                "chain_filter_method": resolved_chain_filter_method,
                "n_events": n_events,
                "mean_lifetime_ps": mean_lifetime,
                "mean_lifetime_std_ps": mean_lifetime_std,
                "median_lifetime_ps": median_lifetime,
                "t_1e_ps": float(t_1e),
                "t_01_ps": float(t_01),
                "contact_cutoff_nm": contact_cutoff_nm,
                "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                "radius_limit_A": radius_limit_A,
                "nac_start": nac_residue_range[0],
                "nac_end": nac_residue_range[1],
                "triplet_size": triplet_size,
            }
        )

        for idx in range(len(t_common)):
            rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "observable": mode_cfg["observable"],
                    "contact_mode": contact_mode,
                    "contact_label": mode_cfg["label"],
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "time_ps": float(t_common[idx]),
                    "survival_mean": float(survival_mean[idx]),
                    "survival_std": float(survival_std[idx]),
                    "t_1e_ps": float(t_1e),
                    "t_01_ps": float(t_01),
                    "mean_lifetime_ps": mean_lifetime,
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                    "radius_limit_A": radius_limit_A,
                    "nac_start": nac_residue_range[0],
                    "nac_end": nac_residue_range[1],
                    "triplet_size": triplet_size,
                }
            )

    df_all = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    traj_summary_df = pd.DataFrame(traj_summary_rows)

    df_all.to_csv(cache_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    traj_summary_df.to_csv(trajectory_summary_csv, index=False)

    print(f"Saved {mode_cfg['label']} survival cache: {cache_csv}")
    print(f"Saved {mode_cfg['label']} summary table: {summary_csv}")
    print(f"Saved {mode_cfg['label']} trajectory table: {trajectory_summary_csv}")
    print(f"\n{mode_cfg['label']} summary metrics:")
    print(summary_df.to_string(index=False))

    return df_all, summary_df


def run_nac_triplet_relaxation_analysis(*args, **kwargs):
    kwargs.setdefault("contact_mode", "triplet_corresponding_nac_bb")
    kwargs.setdefault("cache_csv", NAC_CACHE_CSV)
    kwargs.setdefault("summary_csv", NAC_SUMMARY_CSV)
    kwargs.setdefault("trajectory_summary_csv", NAC_TRAJ_SUMMARY_CSV)
    return run_nac_contact_relaxation_analysis(*args, **kwargs)


def run_nac_contact_relaxation_comparison(
    contact_modes=DEFAULT_CONTACT_MODES,
    combined_summary_csv="nac_contact_mode_relaxation_summary.csv",
    shared_scan=True,
    **kwargs,
):
    output_dir = kwargs.get("output_dir")
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    if shared_scan:
        results, combined_summary = run_nac_contact_relaxation_analysis_multi(
            contact_modes=contact_modes,
            **kwargs,
        )
    else:
        separate_kwargs = dict(kwargs)
        separate_kwargs.pop("use_valid_chain_cache", None)
        separate_kwargs.pop("valid_chain_cache_dir", None)
        results = {}
        summaries = []
        for contact_mode in contact_modes:
            df_all, summary_df = run_nac_contact_relaxation_analysis(
                contact_mode=contact_mode,
                **separate_kwargs,
            )
            results[contact_mode] = {
                "data": df_all,
                "summary": summary_df,
            }
            summaries.append(summary_df)
        combined_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    combined_summary_path = (
        combined_summary_csv
        if os.path.isabs(combined_summary_csv)
        else os.path.join(output_dir, combined_summary_csv)
    )
    combined_summary.to_csv(combined_summary_path, index=False)
    print(f"Saved combined contact-mode summary: {combined_summary_path}")
    return results, combined_summary


def parse_args():
    group_names = [group["name"] for group in DEFAULT_GROUPS]
    parser = argparse.ArgumentParser(
        description="Calculate NAC BB contact survival times for the dense M3IDP trajectories used by dense.ipynb."
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=group_names,
        help="Run only one group. Can be used multiple times. Default: all groups.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=list(CONTACT_MODE_CONFIG),
        help="Contact definition to run. Can be used multiple times. Default: all four NAC contact modes.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "nac_contact_outputs"),
        help="Base directory for per-mode output folders. Default: Analysis/NAC_contact_lifetime/NAC_contact_relaxation_time/nac_contact_outputs.",
    )
    parser.add_argument(
        "--cutoff-nm",
        type=float,
        default=NAC_CONTACT_CUTOFF_NM,
        help="BB-BB contact cutoff in nm. Default: 0.8.",
    )
    parser.add_argument(
        "--radius-limit-nm",
        type=float,
        default=DEFAULT_RADIUS_LIMIT_NM,
        help="Chain COM distance cutoff for --chain-filter-method radius, in nm. Default: 10.0.",
    )
    parser.add_argument(
        "--chain-filter-method",
        choices=("largest_cluster", "radius", "none"),
        default=NAC_CHAIN_FILTER_METHOD,
        help="Valid-chain filter. largest_cluster follows the MSD connected-component definition. Default: largest_cluster.",
    )
    parser.add_argument(
        "--cluster-dist-cutoff-A",
        type=float,
        default=NAC_CLUSTER_DIST_CUTOFF_A,
        help="BB-BB cutoff in Angstrom for largest-cluster detection. Default: 8.0.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=DEFAULT_END_FRAME,
        help="End frame index, or -1 for trajectory end. Default: -1.",
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        default=DEFAULT_SKIP_FRAME,
        help="Trajectory stride. Default: 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress every N analyzed frames. Default: 100.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate even when CSV cache files already exist.",
    )
    parser.add_argument(
        "--separate-modes",
        action="store_true",
        help="Run each contact mode as a separate trajectory scan. Default: shared multi-mode scan.",
    )
    parser.add_argument(
        "--no-valid-chain-cache",
        action="store_true",
        help="Do not reuse the valid-chain/largest-cluster disk cache.",
    )
    parser.add_argument(
        "--valid-chain-cache-dir",
        default=DEFAULT_VALID_CHAIN_CACHE_DIR,
        help="Directory for valid-chain/largest-cluster cache files.",
    )
    parser.add_argument(
        "--no-radius-filter",
        action="store_true",
        help="Legacy alias for --chain-filter-method none.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected_groups = [
        group for group in DEFAULT_GROUPS
        if args.group is None or group["name"] in args.group
    ]
    group_labels = {group["name"]: group["label"] for group in selected_groups}
    group_colors = {group["name"]: group["color"] for group in selected_groups}
    contact_modes = tuple(args.mode) if args.mode is not None else DEFAULT_CONTACT_MODES

    run_nac_contact_relaxation_comparison(
        contact_modes=contact_modes,
        groups=selected_groups,
        group_labels=group_labels,
        group_colors=group_colors,
        radius_limit_A=args.radius_limit_nm * 10.0,
        end_frame=args.end_frame,
        skip_frame=args.skip_frame,
        progress_every=args.progress_every,
        output_dir=args.output_dir,
        contact_cutoff_nm=args.cutoff_nm,
        use_radius_filter=not args.no_radius_filter,
        chain_filter_method="none" if args.no_radius_filter else args.chain_filter_method,
        cluster_dist_cutoff_A=args.cluster_dist_cutoff_A,
        force=args.force,
        shared_scan=not args.separate_modes,
        use_valid_chain_cache=not args.no_valid_chain_cache,
        valid_chain_cache_dir=args.valid_chain_cache_dir,
    )


if __name__ == "__main__":
    main()

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import capped_distance

from nac_triplet_relaxation import (
    BANDED_TRIPLET_DEFINITION,
    CONTACT_MODE_CONFIG,
    DEFAULT_CONTACT_MODES,
    DEFAULT_END_FRAME,
    DEFAULT_GROUP_CHAIN_SELECTIONS,
    DEFAULT_GROUPS,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_RADIUS_LIMIT_NM,
    DEFAULT_SKIP_FRAME,
    DEFAULT_VALID_CHAIN_CACHE_DIR,
    NAC_BACKBONE_NAME,
    NAC_ANTISYMMETRIC_SUM_WINDOW,
    NAC_CHAIN_FILTER_METHOD,
    NAC_CLUSTER_DIST_CUTOFF_A,
    NAC_CONTACT_CUTOFF_NM,
    NAC_PAIR_CHUNK,
    NAC_RESIDUE_RANGE,
    NAC_TRIPLET_SIZE,
    NAC_USE_PBC,
    NAC_USE_RADIUS_FILTER,
    RESIDUES_PER_CHAIN,
    SCRIPT_DIR,
    banded_residue_contacts_from_close_pairs,
    build_banded_antisymmetric_indices,
    build_banded_allowed_matrix,
    build_banded_contact_masks,
    build_chain_bb_indices,
    build_chain_pairs,
    build_chain_selection_mask,
    build_flat_chain_residue_ids,
    build_pair_index_matrix,
    cache_matches_chain_filter,
    crossing_time,
    get_frame_iterator,
    get_valid_box,
    resolve_antisymmetric_sum_target,
    resolve_chain_filter_method,
    select_valid_chains_cached,
)


NAC_ACF_CACHE_CSV = "nac_triplet_antisymmetric_acf_data.csv"
NAC_ACF_SUMMARY_CSV = "nac_triplet_antisymmetric_acf_summary.csv"
NAC_ACF_TRAJ_SUMMARY_CSV = "nac_triplet_antisymmetric_acf_trajectory_summary.csv"

DEFAULT_ACF_KIND = "fluctuation"
DEFAULT_ACF_PAIR_CHUNK = 128


def get_contact_mode_config(contact_mode):
    if contact_mode not in CONTACT_MODE_CONFIG:
        choices = ", ".join(CONTACT_MODE_CONFIG)
        raise ValueError(f"Unknown contact_mode={contact_mode!r}. Choices: {choices}")
    return CONTACT_MODE_CONFIG[contact_mode]


def get_acf_mode_config(contact_mode, acf_kind=DEFAULT_ACF_KIND):
    mode_cfg = dict(get_contact_mode_config(contact_mode))
    mode_cfg["observable"] = mode_cfg["observable"].replace(
        "_survival",
        f"_{acf_kind}_acf",
    )
    mode_cfg["file_prefix"] = f"{mode_cfg['file_prefix']}_{acf_kind}_acf"
    return mode_cfg


def validate_acf_kind(acf_kind):
    choices = {"fluctuation", "conditional"}
    if acf_kind not in choices:
        raise ValueError(f"Unknown acf_kind={acf_kind!r}. Choices: {', '.join(sorted(choices))}")


def next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def autocorrelation_fft_chunk(series, n_lags):
    n_frames = series.shape[1]
    n_fft = next_power_of_two(2 * n_frames - 1)
    fft_values = np.fft.rfft(series, n=n_fft, axis=1)
    corr = np.fft.irfft(fft_values * np.conjugate(fft_values), n=n_fft, axis=1)
    return corr[:, :n_lags].real


def compute_contact_acf(
    contact_series,
    dt_ps,
    acf_kind=DEFAULT_ACF_KIND,
    acf_pair_chunk=DEFAULT_ACF_PAIR_CHUNK,
    max_lag_frames=None,
):
    validate_acf_kind(acf_kind)

    n_pairs, n_frames = contact_series.shape
    if n_pairs == 0 or n_frames < 2:
        return {
            "time_ps": np.array([], dtype=float),
            "acf": np.array([], dtype=float),
            "n_observations": int(n_pairs * n_frames),
            "n_contact_observations": 0,
            "contact_probability": np.nan,
            "n_variable_pairs": 0,
        }

    n_lags = n_frames if max_lag_frames is None else min(int(max_lag_frames) + 1, n_frames)
    lag_indices = np.arange(n_lags, dtype=np.int64)

    contacts_per_pair = np.sum(contact_series, axis=1)
    n_contact_observations = int(np.sum(contacts_per_pair))
    n_observations = int(n_pairs * n_frames)
    contact_probability = float(n_contact_observations / n_observations) if n_observations else np.nan
    n_variable_pairs = int(np.sum((contacts_per_pair > 0) & (contacts_per_pair < n_frames)))

    numerator_sum = np.zeros(n_lags, dtype=np.float64)

    if acf_kind == "conditional":
        denominator_sum = np.zeros(n_lags, dtype=np.float64)

        for pair_start in range(0, n_pairs, acf_pair_chunk):
            pair_end = min(pair_start + acf_pair_chunk, n_pairs)
            chunk = contact_series[pair_start:pair_end].astype(np.float64, copy=False)
            numerator_sum += np.sum(autocorrelation_fft_chunk(chunk, n_lags), axis=0)

            origin_cumsum = np.cumsum(chunk, axis=1)
            denominator_sum += np.sum(origin_cumsum[:, n_frames - 1 - lag_indices], axis=0)

        acf = np.full(n_lags, np.nan, dtype=float)
        valid = denominator_sum > 0.0
        acf[valid] = numerator_sum[valid] / denominator_sum[valid]
    else:
        variance_sum = 0.0

        for pair_start in range(0, n_pairs, acf_pair_chunk):
            pair_end = min(pair_start + acf_pair_chunk, n_pairs)
            chunk = contact_series[pair_start:pair_end].astype(np.float64, copy=False)
            means = np.mean(chunk, axis=1, keepdims=True)
            centered = chunk - means

            numerator_sum += np.sum(autocorrelation_fft_chunk(centered, n_lags), axis=0)
            variance_sum += float(np.sum(centered * centered) / n_frames)

        denominator = (n_frames - lag_indices).astype(np.float64) * variance_sum
        acf = np.full(n_lags, np.nan, dtype=float)
        valid = denominator > 0.0
        acf[valid] = numerator_sum[valid] / denominator[valid]

    return {
        "time_ps": lag_indices.astype(float) * dt_ps,
        "acf": acf,
        "n_observations": n_observations,
        "n_contact_observations": n_contact_observations,
        "contact_probability": contact_probability,
        "n_variable_pairs": n_variable_pairs,
    }


def build_contact_series_for_trajectory(
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
    antisymmetric_sum_target=None,
    antisymmetric_sum_window=NAC_ANTISYMMETRIC_SUM_WINDOW,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    contact_mode="antisymmetric_nac_bb",
    chain_selection=None,
    progress_every=100,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    mode_cfg = get_contact_mode_config(contact_mode)
    print(f"  -> {mode_cfg['label']} ACF: {os.path.basename(xtc_file)}")
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
    if contact_mode == "triplet_antisymmetric_nac_bb" and n_triplets <= 0:
        raise ValueError(f"NAC length {n_nac} is shorter than triplet_size={triplet_size}.")
    band_source_idx, band_partner_idx, resolved_sum_target = build_banded_antisymmetric_indices(
        nac_residue_range,
        antisymmetric_sum_target=antisymmetric_sum_target,
        antisymmetric_sum_window=antisymmetric_sum_window,
    )
    band_allowed = build_banded_allowed_matrix(n_nac, band_source_idx, band_partner_idx)
    n_banded_pairs = len(band_source_idx)
    nac_flat_chain_ids, nac_flat_residue_idx = build_flat_chain_residue_ids(n_chains, n_nac)

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
    contact_series = np.zeros((n_valid_pairs, n_frames_actual), dtype=bool)

    if contact_mode != "any_nac_to_nac_bb" and contact_mode not in {
        "three_antisymmetric_nac_bb",
        "antisymmetric_nac_bb",
        "triplet_antisymmetric_nac_bb",
    }:
        raise ValueError(f"Unsupported contact_mode={contact_mode!r}")

    for frame_idx, ts in enumerate(get_frame_iterator(universe, start_frame, end_frame, skip_frame)):
        if frame_idx % progress_every == 0:
            print(f"     contact frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        nac_positions = universe.atoms.positions[nac_bb_indices]
        box = get_valid_box(ts, use_pbc)
        close_pairs = capped_distance(
            nac_positions.reshape(-1, 3),
            nac_positions.reshape(-1, 3),
            max_cutoff=contact_cutoff_A,
            box=box,
            return_distances=False,
        )

        if contact_mode == "any_nac_to_nac_bb":
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

            contact_series[:, frame_idx] = contact_mask[valid_pair_indices]
        else:
            needs_triplet_contacts = contact_mode == "triplet_antisymmetric_nac_bb"
            contact_result = banded_residue_contacts_from_close_pairs(
                close_pairs,
                nac_flat_chain_ids,
                nac_flat_residue_idx,
                pair_index,
                full_to_valid_pair,
                valid_chain_mask,
                band_allowed,
                n_valid_pairs,
                n_nac,
                return_pair_contacts=needs_triplet_contacts,
            )
            if needs_triplet_contacts:
                residue_contacts, residue_pair_contacts = contact_result
            else:
                residue_contacts = contact_result
                residue_pair_contacts = None
            contact_mask = build_banded_contact_masks(
                residue_contacts,
                (contact_mode,),
                n_triplets,
                triplet_size=triplet_size,
                residue_pair_contacts=residue_pair_contacts,
            )[contact_mode]
            contact_series[:, frame_idx] = contact_mask

    print(f"     contact frame {n_frames_actual:>5}/{n_frames_actual}")
    return {
        "contact_series": contact_series,
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
        "n_banded_residue_pairs": int(n_banded_pairs),
        "contact_mode": contact_mode,
        "contact_label": mode_cfg["label"],
        "event_unit": mode_cfg["event_unit"],
        "contact_cutoff_nm": float(contact_cutoff_nm),
        "antisymmetric_sum_target": float(resolved_sum_target),
        "antisymmetric_sum_window": float(antisymmetric_sum_window),
    }


def build_contact_series_for_trajectory_multi(
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
    antisymmetric_sum_target=None,
    antisymmetric_sum_window=NAC_ANTISYMMETRIC_SUM_WINDOW,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    chain_selection=None,
    progress_every=100,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    contact_modes = tuple(dict.fromkeys(contact_modes))
    for contact_mode in contact_modes:
        get_contact_mode_config(contact_mode)

    print(
        "  -> shared ACF contact-series scan for "
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
    if "triplet_antisymmetric_nac_bb" in contact_modes and n_triplets <= 0:
        raise ValueError(f"NAC length {n_nac} is shorter than triplet_size={triplet_size}.")
    band_source_idx, band_partner_idx, resolved_sum_target = build_banded_antisymmetric_indices(
        nac_residue_range,
        antisymmetric_sum_target=antisymmetric_sum_target,
        antisymmetric_sum_window=antisymmetric_sum_window,
    )
    band_allowed = build_banded_allowed_matrix(n_nac, band_source_idx, band_partner_idx)
    n_banded_pairs = len(band_source_idx)
    nac_flat_chain_ids, nac_flat_residue_idx = build_flat_chain_residue_ids(n_chains, n_nac)

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
    contact_series_by_mode = {
        contact_mode: np.zeros((n_valid_pairs, n_frames_actual), dtype=bool)
        for contact_mode in contact_modes
    }
    mem_gib = (n_valid_pairs * n_frames_actual * len(contact_modes)) / (1024 ** 3)
    print(f"     contact-series memory estimate: {mem_gib:.2f} GiB")

    needs_any = "any_nac_to_nac_bb" in contact_modes
    antisymmetric_modes = tuple(mode for mode in contact_modes if mode != "any_nac_to_nac_bb")

    for frame_idx, ts in enumerate(get_frame_iterator(universe, start_frame, end_frame, skip_frame)):
        if frame_idx % progress_every == 0:
            print(f"     contact frame {frame_idx:>5}/{n_frames_actual}", end="\r")

        nac_positions = universe.atoms.positions[nac_bb_indices]
        box = get_valid_box(ts, use_pbc)
        close_pairs = capped_distance(
            nac_positions.reshape(-1, 3),
            nac_positions.reshape(-1, 3),
            max_cutoff=contact_cutoff_A,
            box=box,
            return_distances=False,
        )

        if needs_any:
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

            contact_series_by_mode["any_nac_to_nac_bb"][:, frame_idx] = contact_mask[valid_pair_indices]

        if antisymmetric_modes:
            needs_triplet_contacts = "triplet_antisymmetric_nac_bb" in antisymmetric_modes
            contact_result = banded_residue_contacts_from_close_pairs(
                close_pairs,
                nac_flat_chain_ids,
                nac_flat_residue_idx,
                pair_index,
                full_to_valid_pair,
                valid_chain_mask,
                band_allowed,
                n_valid_pairs,
                n_nac,
                return_pair_contacts=needs_triplet_contacts,
            )
            if needs_triplet_contacts:
                residue_contacts, residue_pair_contacts = contact_result
            else:
                residue_contacts = contact_result
                residue_pair_contacts = None
            masks = build_banded_contact_masks(
                residue_contacts,
                antisymmetric_modes,
                n_triplets,
                triplet_size=triplet_size,
                residue_pair_contacts=residue_pair_contacts,
            )

            for contact_mode, contact_mask in masks.items():
                contact_series_by_mode[contact_mode][:, frame_idx] = contact_mask

    print(f"     contact frame {n_frames_actual:>5}/{n_frames_actual}")

    results = {}
    for contact_mode in contact_modes:
        mode_cfg = get_contact_mode_config(contact_mode)
        results[contact_mode] = {
            "contact_series": contact_series_by_mode[contact_mode],
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
            "n_banded_residue_pairs": int(n_banded_pairs),
            "contact_mode": contact_mode,
            "contact_label": mode_cfg["label"],
            "event_unit": mode_cfg["event_unit"],
            "contact_cutoff_nm": float(contact_cutoff_nm),
            "antisymmetric_sum_target": float(resolved_sum_target),
            "antisymmetric_sum_window": float(antisymmetric_sum_window),
        }

    return results


def analyze_single_nac_contact_acf_trajectory(
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
    antisymmetric_sum_target=None,
    antisymmetric_sum_window=NAC_ANTISYMMETRIC_SUM_WINDOW,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    contact_mode="antisymmetric_nac_bb",
    chain_selection=None,
    progress_every=100,
    acf_kind=DEFAULT_ACF_KIND,
    acf_pair_chunk=DEFAULT_ACF_PAIR_CHUNK,
    max_lag_frames=None,
    max_lag_ns=None,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    contact_data = build_contact_series_for_trajectory(
        tpr_file,
        xtc_file,
        start_frame,
        end_frame,
        skip_frame,
        radius_limit_A,
        nac_residue_range=nac_residue_range,
        residues_per_chain=residues_per_chain,
        triplet_size=triplet_size,
        backbone_name=backbone_name,
        contact_cutoff_nm=contact_cutoff_nm,
        antisymmetric_sum_target=antisymmetric_sum_target,
        antisymmetric_sum_window=antisymmetric_sum_window,
        pair_chunk=pair_chunk,
        use_pbc=use_pbc,
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
        cluster_dist_cutoff_A=cluster_dist_cutoff_A,
        contact_mode=contact_mode,
        chain_selection=chain_selection,
        progress_every=progress_every,
        use_valid_chain_cache=use_valid_chain_cache,
        valid_chain_cache_dir=valid_chain_cache_dir,
    )

    if max_lag_frames is None and max_lag_ns is not None:
        max_lag_frames = int(np.floor(max_lag_ns * 1000.0 / contact_data["dt_ps"]))

    acf = compute_contact_acf(
        contact_data["contact_series"],
        contact_data["dt_ps"],
        acf_kind=acf_kind,
        acf_pair_chunk=acf_pair_chunk,
        max_lag_frames=max_lag_frames,
    )
    del contact_data["contact_series"]

    acf.update(contact_data)
    acf["acf_kind"] = acf_kind
    return acf


def analyze_single_nac_triplet_acf_trajectory(*args, **kwargs):
    kwargs["contact_mode"] = "triplet_antisymmetric_nac_bb"
    return analyze_single_nac_contact_acf_trajectory(*args, **kwargs)


def build_acf_stack(time_list, acf_list, dt_hint):
    t_max = max(t[-1] for t in time_list)
    dt_common = max(
        np.median(np.diff(t)) if len(t) > 1 else dt_hint
        for t in time_list
    )
    npts = int(np.floor(t_max / dt_common)) + 1
    t_common = np.arange(npts, dtype=float) * dt_common
    acf_stack = np.array(
        [
            np.interp(t_common, t, acf, left=1.0, right=np.nan)
            for t, acf in zip(time_list, acf_list)
        ]
    )
    return t_common, acf_stack


def resolve_acf_output_paths(output_dir, contact_mode, acf_kind=DEFAULT_ACF_KIND):
    mode_cfg = get_acf_mode_config(contact_mode, acf_kind=acf_kind)
    mode_output_dir = os.path.join(output_dir, mode_cfg["file_prefix"])
    return {
        "mode_output_dir": mode_output_dir,
        "cache_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_data.csv"),
        "summary_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_summary.csv"),
        "trajectory_summary_csv": os.path.join(mode_output_dir, f"{mode_cfg['file_prefix']}_trajectory_summary.csv"),
    }


def run_nac_contact_acf_analysis_multi(
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
    antisymmetric_sum_target=None,
    antisymmetric_sum_window=NAC_ANTISYMMETRIC_SUM_WINDOW,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    force=False,
    acf_kind=DEFAULT_ACF_KIND,
    acf_pair_chunk=DEFAULT_ACF_PAIR_CHUNK,
    max_lag_ns=None,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    validate_acf_kind(acf_kind)
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
        paths = resolve_acf_output_paths(output_dir, contact_mode, acf_kind=acf_kind)
        if not force and os.path.exists(paths["cache_csv"]) and os.path.exists(paths["summary_csv"]):
            print(f"Detected cached {contact_mode} {acf_kind} ACF data; loading.")
            df_all = pd.read_csv(paths["cache_csv"])
            summary_df = pd.read_csv(paths["summary_csv"])
            cache_filter_matches = cache_matches_chain_filter(
                summary_df,
                resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                radius_limit_A=radius_limit_A,
            )
            if cache_filter_matches and max_lag_ns is not None and not df_all.empty:
                cached_max_lag_ps = float(df_all["time_ps"].max())
                required_max_lag_ps = float(max_lag_ns) * 1e3
                if cached_max_lag_ps + 1e-9 >= required_max_lag_ps:
                    results[contact_mode] = {"data": df_all, "summary": summary_df}
                    continue
                print(
                    f"Cached {contact_mode} ACF reaches {cached_max_lag_ps / 1e3:.3f} ns, "
                    f"but {max_lag_ns:.3f} ns was requested; recalculating."
                )
            elif cache_filter_matches and max_lag_ns is None:
                results[contact_mode] = {"data": df_all, "summary": summary_df}
                continue
            elif cache_filter_matches and max_lag_ns is not None and df_all.empty:
                print(f"Cached {contact_mode} ACF table is empty; recalculating.")
            else:
                print(f"Cached {contact_mode} ACF data use different chain filtering or contact definition; recalculating.")
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
        print(f"Processing shared antisymmetric {acf_kind} ACF scan: {group_label}")
        group_chain_selection = group.get(
            "chain_selection",
            DEFAULT_GROUP_CHAIN_SELECTIONS.get(group_name, "all"),
        )
        print(f"Chain selection: {group_chain_selection}")

        time_lists = {mode: [] for mode in modes_to_run}
        acf_lists = {mode: [] for mode in modes_to_run}
        dt_hints = {mode: [] for mode in modes_to_run}
        traj_contact_observations = {mode: [] for mode in modes_to_run}
        traj_total_observations = {mode: [] for mode in modes_to_run}
        traj_variable_pairs = {mode: [] for mode in modes_to_run}

        for traj_idx, traj in enumerate(group["trajectories"], start=1):
            contact_data_by_mode = build_contact_series_for_trajectory_multi(
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
                antisymmetric_sum_target=antisymmetric_sum_target,
                antisymmetric_sum_window=antisymmetric_sum_window,
                pair_chunk=pair_chunk,
                use_pbc=use_pbc,
                use_radius_filter=use_radius_filter,
                chain_filter_method=resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                chain_selection=traj.get("chain_selection", group_chain_selection),
                progress_every=progress_every,
                use_valid_chain_cache=use_valid_chain_cache,
                valid_chain_cache_dir=valid_chain_cache_dir,
            )

            for contact_mode in modes_to_run:
                contact_data = contact_data_by_mode[contact_mode]
                max_lag_frames = None
                if max_lag_ns is not None:
                    max_lag_frames = int(np.floor(max_lag_ns * 1000.0 / contact_data["dt_ps"]))
                acf = compute_contact_acf(
                    contact_data["contact_series"],
                    contact_data["dt_ps"],
                    acf_kind=acf_kind,
                    acf_pair_chunk=acf_pair_chunk,
                    max_lag_frames=max_lag_frames,
                )
                del contact_data["contact_series"]
                acf.update(contact_data)
                acf["acf_kind"] = acf_kind

                if len(acf["time_ps"]) == 0 or not np.any(np.isfinite(acf["acf"])):
                    traj_t_1e = np.nan
                    traj_t_01 = np.nan
                else:
                    traj_t_1e = crossing_time(acf["time_ps"], acf["acf"], 1.0 / np.e)
                    traj_t_01 = crossing_time(acf["time_ps"], acf["acf"], 0.1)
                traj_summary_rows_by_mode[contact_mode].append(
                    {
                        "group": group_name,
                        "group_label": group_label,
                        "trajectory": traj_idx,
                        "xtc": traj["xtc"],
                        "acf_kind": acf_kind,
                        "n_observations": acf["n_observations"],
                        "n_contact_observations": acf["n_contact_observations"],
                        "contact_probability": acf["contact_probability"],
                        "n_variable_pairs": acf["n_variable_pairs"],
                        "t_1e_ps": float(traj_t_1e),
                        "t_01_ps": float(traj_t_01),
                        "dt_ps": acf["dt_ps"],
                        "n_frames": acf["n_frames"],
                        "n_chains": acf["n_chains"],
                        "chain_selection": acf["chain_selection"],
                        "n_selected_chains": acf["n_selected_chains"],
                        "selected_chain_start": acf["selected_chain_start"],
                        "selected_chain_end": acf["selected_chain_end"],
                        "chain_filter_method": acf["chain_filter_method"],
                        "cluster_dist_cutoff_A": acf["cluster_dist_cutoff_A"],
                        "radius_limit_A": acf["radius_limit_A"],
                        "n_valid_chains": acf["n_valid_chains"],
                        "n_pairs": acf["n_pairs"],
                        "n_valid_pairs": acf["n_valid_pairs"],
                        "n_contact_sites_per_pair": acf["n_contact_sites_per_pair"],
                        "n_triplets_per_pair": acf["n_triplets_per_pair"],
                        "n_banded_residue_pairs": acf["n_banded_residue_pairs"],
                        "contact_mode": acf["contact_mode"],
                        "contact_label": acf["contact_label"],
                        "event_unit": acf["event_unit"],
                        "contact_cutoff_nm": acf["contact_cutoff_nm"],
                        "antisymmetric_sum_target": acf["antisymmetric_sum_target"],
                        "antisymmetric_sum_window": acf["antisymmetric_sum_window"],
                    }
                )

                mode_cfg = get_acf_mode_config(contact_mode, acf_kind=acf_kind)
                if len(acf["time_ps"]) == 0 or not np.any(np.isfinite(acf["acf"])):
                    print(f"     no usable {mode_cfg['label']} {acf_kind} ACF")
                    continue

                print(
                    f"     {mode_cfg['label']}: contact_probability={acf['contact_probability']:.6f}, "
                    f"contact_observations={acf['n_contact_observations']}, "
                    f"variable_pairs={acf['n_variable_pairs']}"
                )

                time_lists[contact_mode].append(acf["time_ps"])
                acf_lists[contact_mode].append(acf["acf"])
                dt_hints[contact_mode].append(acf["dt_ps"])
                traj_contact_observations[contact_mode].append(acf["n_contact_observations"])
                traj_total_observations[contact_mode].append(acf["n_observations"])
                traj_variable_pairs[contact_mode].append(acf["n_variable_pairs"])

            del contact_data_by_mode

        for contact_mode in modes_to_run:
            mode_cfg = get_acf_mode_config(contact_mode, acf_kind=acf_kind)
            if not time_lists[contact_mode]:
                summary_rows_by_mode[contact_mode].append(
                    {
                        "group": group_name,
                        "group_label": group_label,
                        "observable": mode_cfg["observable"],
                        "contact_mode": contact_mode,
                        "contact_label": mode_cfg["label"],
                        "acf_kind": acf_kind,
                        "event_unit": mode_cfg["event_unit"],
                        "chain_selection": group_chain_selection,
                        "chain_filter_method": resolved_chain_filter_method,
                        "n_observations": 0,
                        "n_contact_observations": 0,
                        "contact_probability": np.nan,
                        "n_variable_pairs": 0,
                        "t_1e_ps": np.nan,
                        "t_01_ps": np.nan,
                        "contact_cutoff_nm": contact_cutoff_nm,
                        "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                            nac_residue_range,
                            antisymmetric_sum_target=antisymmetric_sum_target,
                        ),
                        "antisymmetric_sum_window": antisymmetric_sum_window,
                        "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                        "radius_limit_A": radius_limit_A,
                        "nac_start": nac_residue_range[0],
                        "nac_end": nac_residue_range[1],
                        "triplet_size": triplet_size,
                    }
                )
                continue

            t_common, acf_stack = build_acf_stack(
                time_lists[contact_mode],
                acf_lists[contact_mode],
                dt_hint=max(dt_hints[contact_mode]),
            )
            acf_mean = np.nanmean(acf_stack, axis=0)
            acf_std = np.nanstd(acf_stack, axis=0)

            n_contact_observations = int(np.sum(traj_contact_observations[contact_mode]))
            n_observations = int(np.sum(traj_total_observations[contact_mode]))
            contact_probability = float(n_contact_observations / n_observations) if n_observations else np.nan
            n_variable_pairs = int(np.sum(traj_variable_pairs[contact_mode]))

            t_1e = crossing_time(t_common, acf_mean, 1.0 / np.e)
            t_01 = crossing_time(t_common, acf_mean, 0.1)

            print(
                f"  [{mode_cfg['label']} {acf_kind} ACF] "
                f"t(1/e)={t_1e:.2f} ps, t(0.1)={t_01:.2f} ps, "
                f"contact_probability={contact_probability:.6f}"
            )

            summary_rows_by_mode[contact_mode].append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "observable": mode_cfg["observable"],
                    "contact_mode": contact_mode,
                    "contact_label": mode_cfg["label"],
                    "acf_kind": acf_kind,
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "n_observations": n_observations,
                    "n_contact_observations": n_contact_observations,
                    "contact_probability": contact_probability,
                    "n_variable_pairs": n_variable_pairs,
                    "t_1e_ps": float(t_1e),
                    "t_01_ps": float(t_01),
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                        nac_residue_range,
                        antisymmetric_sum_target=antisymmetric_sum_target,
                    ),
                    "antisymmetric_sum_window": antisymmetric_sum_window,
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
                        "acf_kind": acf_kind,
                        "event_unit": mode_cfg["event_unit"],
                        "chain_selection": group_chain_selection,
                        "chain_filter_method": resolved_chain_filter_method,
                        "time_ps": float(t_common[idx]),
                        "acf_mean": float(acf_mean[idx]),
                        "acf_std": float(acf_std[idx]),
                        "t_1e_ps": float(t_1e),
                        "t_01_ps": float(t_01),
                        "contact_probability": contact_probability,
                        "contact_cutoff_nm": contact_cutoff_nm,
                        "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                            nac_residue_range,
                            antisymmetric_sum_target=antisymmetric_sum_target,
                        ),
                        "antisymmetric_sum_window": antisymmetric_sum_window,
                        "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                        "radius_limit_A": radius_limit_A,
                        "nac_start": nac_residue_range[0],
                        "nac_end": nac_residue_range[1],
                        "triplet_size": triplet_size,
                    }
                )

    for contact_mode in modes_to_run:
        paths = resolve_acf_output_paths(output_dir, contact_mode, acf_kind=acf_kind)
        os.makedirs(paths["mode_output_dir"], exist_ok=True)

        df_all = pd.DataFrame(rows_by_mode[contact_mode])
        summary_df = pd.DataFrame(summary_rows_by_mode[contact_mode])
        traj_summary_df = pd.DataFrame(traj_summary_rows_by_mode[contact_mode])
        for table in (df_all, summary_df, traj_summary_df):
            if not table.empty:
                table["banded_triplet_definition"] = BANDED_TRIPLET_DEFINITION

        df_all.to_csv(paths["cache_csv"], index=False)
        summary_df.to_csv(paths["summary_csv"], index=False)
        traj_summary_df.to_csv(paths["trajectory_summary_csv"], index=False)

        mode_cfg = get_acf_mode_config(contact_mode, acf_kind=acf_kind)
        print(f"Saved {mode_cfg['label']} {acf_kind} ACF cache: {paths['cache_csv']}")
        print(f"Saved {mode_cfg['label']} {acf_kind} ACF summary table: {paths['summary_csv']}")
        print(f"Saved {mode_cfg['label']} {acf_kind} ACF trajectory table: {paths['trajectory_summary_csv']}")
        results[contact_mode] = {"data": df_all, "summary": summary_df}

    combined_summary = pd.concat(
        [results[mode]["summary"] for mode in contact_modes if mode in results],
        ignore_index=True,
    )
    return results, combined_summary


def run_nac_contact_acf_analysis(
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
    antisymmetric_sum_target=None,
    antisymmetric_sum_window=NAC_ANTISYMMETRIC_SUM_WINDOW,
    pair_chunk=NAC_PAIR_CHUNK,
    use_pbc=NAC_USE_PBC,
    use_radius_filter=NAC_USE_RADIUS_FILTER,
    chain_filter_method=NAC_CHAIN_FILTER_METHOD,
    cluster_dist_cutoff_A=NAC_CLUSTER_DIST_CUTOFF_A,
    contact_mode="antisymmetric_nac_bb",
    force=False,
    acf_kind=DEFAULT_ACF_KIND,
    acf_pair_chunk=DEFAULT_ACF_PAIR_CHUNK,
    max_lag_ns=None,
    use_valid_chain_cache=True,
    valid_chain_cache_dir=None,
):
    validate_acf_kind(acf_kind)
    group_labels = group_labels or {group["name"]: group.get("label", group["name"]) for group in groups}
    group_colors = group_colors or {group["name"]: group.get("color", "#1f77b4") for group in groups}
    mode_cfg = get_acf_mode_config(contact_mode, acf_kind=acf_kind)
    resolved_chain_filter_method = resolve_chain_filter_method(
        use_radius_filter=use_radius_filter,
        chain_filter_method=chain_filter_method,
    )

    if cache_csv is None:
        cache_csv = f"{mode_cfg['file_prefix']}_data.csv"
    if summary_csv is None:
        summary_csv = f"{mode_cfg['file_prefix']}_summary.csv"
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
        print(f"Detected {cache_csv} and {summary_csv}, load cached NAC ACF data...")
        df_all = pd.read_csv(cache_csv)
        summary_df = pd.read_csv(summary_csv)
        cache_filter_matches = cache_matches_chain_filter(
            summary_df,
            resolved_chain_filter_method,
            cluster_dist_cutoff_A=cluster_dist_cutoff_A,
            radius_limit_A=radius_limit_A,
        )
        if not cache_filter_matches:
            print("Cached NAC ACF data use different chain filtering or contact definition; recalculating.")
        elif max_lag_ns is not None and not df_all.empty:
            cached_max_lag_ps = float(df_all["time_ps"].max())
            required_max_lag_ps = float(max_lag_ns) * 1e3
            if cached_max_lag_ps + 1e-9 < required_max_lag_ps:
                print(
                    f"Cached ACF reaches {cached_max_lag_ps / 1e3:.3f} ns, "
                    f"but {max_lag_ns:.3f} ns was requested; recalculating."
                )
            else:
                return df_all, summary_df
        elif cache_filter_matches and max_lag_ns is not None and df_all.empty:
            print("Cached ACF table is empty; recalculating.")
        elif cache_filter_matches:
            return df_all, summary_df

    rows = []
    summary_rows = []
    traj_summary_rows = []

    for group in groups:
        group_name = group["name"]
        group_label = group_labels.get(group_name, group.get("label", group_name))

        print("=" * 80)
        print(f"Processing {mode_cfg['label']} {acf_kind} ACF: {group_label}")
        group_chain_selection = group.get(
            "chain_selection",
            DEFAULT_GROUP_CHAIN_SELECTIONS.get(group_name, "all"),
        )
        print(f"Chain selection: {group_chain_selection}")

        time_list = []
        acf_list = []
        dt_hints = []
        traj_contact_observations = []
        traj_total_observations = []
        traj_variable_pairs = []

        for traj_idx, traj in enumerate(group["trajectories"], start=1):
            result = analyze_single_nac_contact_acf_trajectory(
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
                antisymmetric_sum_target=antisymmetric_sum_target,
                antisymmetric_sum_window=antisymmetric_sum_window,
                pair_chunk=pair_chunk,
                use_pbc=use_pbc,
                use_radius_filter=use_radius_filter,
                chain_filter_method=resolved_chain_filter_method,
                cluster_dist_cutoff_A=cluster_dist_cutoff_A,
                contact_mode=contact_mode,
                chain_selection=traj.get("chain_selection", group_chain_selection),
                progress_every=progress_every,
                acf_kind=acf_kind,
                acf_pair_chunk=acf_pair_chunk,
                max_lag_ns=max_lag_ns,
                use_valid_chain_cache=use_valid_chain_cache,
                valid_chain_cache_dir=valid_chain_cache_dir,
            )

            if len(result["time_ps"]) == 0 or not np.any(np.isfinite(result["acf"])):
                traj_t_1e = np.nan
                traj_t_01 = np.nan
            else:
                traj_t_1e = crossing_time(result["time_ps"], result["acf"], 1.0 / np.e)
                traj_t_01 = crossing_time(result["time_ps"], result["acf"], 0.1)
            traj_summary_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "trajectory": traj_idx,
                    "xtc": traj["xtc"],
                    "acf_kind": acf_kind,
                    "n_observations": result["n_observations"],
                    "n_contact_observations": result["n_contact_observations"],
                    "contact_probability": result["contact_probability"],
                    "n_variable_pairs": result["n_variable_pairs"],
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
                    "n_banded_residue_pairs": result["n_banded_residue_pairs"],
                    "contact_mode": result["contact_mode"],
                    "contact_label": result["contact_label"],
                    "event_unit": result["event_unit"],
                    "contact_cutoff_nm": result["contact_cutoff_nm"],
                    "antisymmetric_sum_target": result["antisymmetric_sum_target"],
                    "antisymmetric_sum_window": result["antisymmetric_sum_window"],
                }
            )

            if len(result["time_ps"]) == 0 or not np.any(np.isfinite(result["acf"])):
                print(f"     no usable {mode_cfg['label']} {acf_kind} ACF")
                continue

            print(
                f"     contact_probability={result['contact_probability']:.6f}, "
                f"contact_observations={result['n_contact_observations']}, "
                f"variable_pairs={result['n_variable_pairs']}"
            )

            time_list.append(result["time_ps"])
            acf_list.append(result["acf"])
            dt_hints.append(result["dt_ps"])
            traj_contact_observations.append(result["n_contact_observations"])
            traj_total_observations.append(result["n_observations"])
            traj_variable_pairs.append(result["n_variable_pairs"])

        if not time_list:
            summary_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "observable": mode_cfg["observable"],
                    "contact_mode": contact_mode,
                    "contact_label": mode_cfg["label"],
                    "acf_kind": acf_kind,
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "n_observations": 0,
                    "n_contact_observations": 0,
                    "contact_probability": np.nan,
                    "n_variable_pairs": 0,
                    "t_1e_ps": np.nan,
                    "t_01_ps": np.nan,
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                        nac_residue_range,
                        antisymmetric_sum_target=antisymmetric_sum_target,
                    ),
                    "antisymmetric_sum_window": antisymmetric_sum_window,
                    "cluster_dist_cutoff_A": cluster_dist_cutoff_A,
                    "radius_limit_A": radius_limit_A,
                    "nac_start": nac_residue_range[0],
                    "nac_end": nac_residue_range[1],
                    "triplet_size": triplet_size,
                }
            )
            continue

        t_common, acf_stack = build_acf_stack(time_list, acf_list, dt_hint=max(dt_hints))
        acf_mean = np.nanmean(acf_stack, axis=0)
        acf_std = np.nanstd(acf_stack, axis=0)

        n_contact_observations = int(np.sum(traj_contact_observations))
        n_observations = int(np.sum(traj_total_observations))
        contact_probability = float(n_contact_observations / n_observations) if n_observations else np.nan
        n_variable_pairs = int(np.sum(traj_variable_pairs))

        t_1e = crossing_time(t_common, acf_mean, 1.0 / np.e)
        t_01 = crossing_time(t_common, acf_mean, 0.1)

        print(
            f"  [{mode_cfg['label']} {acf_kind} ACF] "
            f"t(1/e)={t_1e:.2f} ps, t(0.1)={t_01:.2f} ps, "
            f"contact_probability={contact_probability:.6f}"
        )

        summary_rows.append(
            {
                "group": group_name,
                "group_label": group_label,
                "observable": mode_cfg["observable"],
                "contact_mode": contact_mode,
                "contact_label": mode_cfg["label"],
                "acf_kind": acf_kind,
                "event_unit": mode_cfg["event_unit"],
                "chain_selection": group_chain_selection,
                "chain_filter_method": resolved_chain_filter_method,
                "n_observations": n_observations,
                "n_contact_observations": n_contact_observations,
                "contact_probability": contact_probability,
                "n_variable_pairs": n_variable_pairs,
                "t_1e_ps": float(t_1e),
                "t_01_ps": float(t_01),
                "contact_cutoff_nm": contact_cutoff_nm,
                "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                    nac_residue_range,
                    antisymmetric_sum_target=antisymmetric_sum_target,
                ),
                "antisymmetric_sum_window": antisymmetric_sum_window,
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
                    "acf_kind": acf_kind,
                    "event_unit": mode_cfg["event_unit"],
                    "chain_selection": group_chain_selection,
                    "chain_filter_method": resolved_chain_filter_method,
                    "time_ps": float(t_common[idx]),
                    "acf_mean": float(acf_mean[idx]),
                    "acf_std": float(acf_std[idx]),
                    "t_1e_ps": float(t_1e),
                    "t_01_ps": float(t_01),
                    "contact_probability": contact_probability,
                    "contact_cutoff_nm": contact_cutoff_nm,
                    "antisymmetric_sum_target": resolve_antisymmetric_sum_target(
                        nac_residue_range,
                        antisymmetric_sum_target=antisymmetric_sum_target,
                    ),
                    "antisymmetric_sum_window": antisymmetric_sum_window,
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
    for table in (df_all, summary_df, traj_summary_df):
        if not table.empty:
            table["banded_triplet_definition"] = BANDED_TRIPLET_DEFINITION

    df_all.to_csv(cache_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    traj_summary_df.to_csv(trajectory_summary_csv, index=False)

    print(f"Saved {mode_cfg['label']} {acf_kind} ACF cache: {cache_csv}")
    print(f"Saved {mode_cfg['label']} {acf_kind} ACF summary table: {summary_csv}")
    print(f"Saved {mode_cfg['label']} {acf_kind} ACF trajectory table: {trajectory_summary_csv}")
    print(f"\n{mode_cfg['label']} {acf_kind} ACF summary metrics:")
    print(summary_df.to_string(index=False))

    return df_all, summary_df


def run_nac_triplet_acf_analysis(*args, **kwargs):
    kwargs.setdefault("contact_mode", "triplet_antisymmetric_nac_bb")
    kwargs.setdefault("cache_csv", NAC_ACF_CACHE_CSV)
    kwargs.setdefault("summary_csv", NAC_ACF_SUMMARY_CSV)
    kwargs.setdefault("trajectory_summary_csv", NAC_ACF_TRAJ_SUMMARY_CSV)
    return run_nac_contact_acf_analysis(*args, **kwargs)


def run_nac_contact_acf_comparison(
    contact_modes=DEFAULT_CONTACT_MODES,
    combined_summary_csv="nac_contact_mode_acf_summary.csv",
    shared_scan=True,
    **kwargs,
):
    output_dir = kwargs.get("output_dir")
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    if shared_scan and len(tuple(contact_modes)) > 1:
        results, combined_summary = run_nac_contact_acf_analysis_multi(
            contact_modes=contact_modes,
            **kwargs,
        )
        combined_summary_path = (
            combined_summary_csv
            if os.path.isabs(combined_summary_csv)
            else os.path.join(output_dir, combined_summary_csv)
        )
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"Saved combined contact-mode ACF summary: {combined_summary_path}")
        return results, combined_summary

    results = {}
    summaries = []
    for contact_mode in contact_modes:
        df_all, summary_df = run_nac_contact_acf_analysis(
            contact_mode=contact_mode,
            **kwargs,
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
    print(f"Saved combined contact-mode ACF summary: {combined_summary_path}")
    return results, combined_summary


def parse_args():
    group_names = [group["name"] for group in DEFAULT_GROUPS]
    parser = argparse.ArgumentParser(
        description="Calculate antisymmetric NAC BB contact autocorrelation functions for the dense M3IDP trajectories."
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
        help="Contact definition to run. Can be used multiple times. Default: all antisymmetric NAC contact modes.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "nac_contact_acf_outputs"),
        help="Base directory for per-mode output folders. Default: this folder/nac_contact_acf_outputs.",
    )
    parser.add_argument(
        "--cutoff-nm",
        type=float,
        default=NAC_CONTACT_CUTOFF_NM,
        help="BB-BB contact cutoff in nm. Default: 0.8.",
    )
    parser.add_argument(
        "--antisymmetric-sum-target",
        type=float,
        default=None,
        help="Residue-number sum for the banded antisymmetric diagonal. Default: residue_range_start + residue_range_end.",
    )
    parser.add_argument(
        "--antisymmetric-sum-window",
        type=float,
        default=NAC_ANTISYMMETRIC_SUM_WINDOW,
        help="Require abs(resid_i + resid_j - target) < this value. Default: 8.",
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
        "--acf-kind",
        choices=("fluctuation", "conditional"),
        default=DEFAULT_ACF_KIND,
        help="ACF definition. fluctuation subtracts the mean and decays to 0; conditional is <h(0)h(t)>/<h(0)>.",
    )
    parser.add_argument(
        "--acf-pair-chunk",
        type=int,
        default=DEFAULT_ACF_PAIR_CHUNK,
        help="Number of chain-pair time series per FFT chunk. Default: 128.",
    )
    parser.add_argument(
        "--max-lag-ns",
        type=float,
        default=None,
        help="Maximum ACF lag time to write, in ns. Default: full trajectory length.",
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

    run_nac_contact_acf_comparison(
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
        antisymmetric_sum_target=args.antisymmetric_sum_target,
        antisymmetric_sum_window=args.antisymmetric_sum_window,
        use_radius_filter=not args.no_radius_filter,
        chain_filter_method="none" if args.no_radius_filter else args.chain_filter_method,
        cluster_dist_cutoff_A=args.cluster_dist_cutoff_A,
        force=args.force,
        acf_kind=args.acf_kind,
        acf_pair_chunk=args.acf_pair_chunk,
        max_lag_ns=args.max_lag_ns,
        shared_scan=not args.separate_modes,
        use_valid_chain_cache=not args.no_valid_chain_cache,
        valid_chain_cache_dir=args.valid_chain_cache_dir,
    )


if __name__ == "__main__":
    main()

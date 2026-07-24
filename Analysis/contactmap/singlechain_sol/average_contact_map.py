import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import argparse
import os

def tanh_switching(dist_matrix, d0, beta):
    """平滑 switching function，输出从1平滑过渡到0"""
    return 0.5 * (1 - np.tanh(beta * (dist_matrix - d0)))

def compute_contact_map(universe, selection='name BB', cutoff=8, beta=1):
    """
    计算所有帧的 contact map，并返回平均 contact 矩阵
    """
    bb_atoms = universe.select_atoms(selection)
    n_atoms = len(bb_atoms)
    contact_sum = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    n_frames = 0

    for ts in universe.trajectory:
        positions = bb_atoms.positions
        dist_matrix = distance_array(positions, positions)
        contact_map = tanh_switching(dist_matrix, d0=cutoff, beta=beta)

        np.fill_diagonal(contact_map, 1)  # 自身设为1
        contact_sum += contact_map
        n_frames += 1

    contact_avg = contact_sum / n_frames
    return contact_avg

def main():
    parser = argparse.ArgumentParser(description="Compute average contact map of BB atoms from .xtc file.")
    parser.add_argument("-s", "--structure", required=True, help="Structure file (.gro or .pdb)")
    parser.add_argument("-f", "--trajectory", required=True, help="Trajectory file (.xtc)")
    parser.add_argument("-o", "--output", default="contact_map_avg.npy", help="Output file name for average contact matrix")
    parser.add_argument("--cutoff", type=float, default=8, help="Cutoff distance in nm (default: 8)")
    parser.add_argument("--beta", type=float, default=1, help="Sharpness parameter for smooth switching function (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.structure) or not os.path.exists(args.trajectory):
        raise FileNotFoundError("Structure or trajectory file not found.")

    print("Loading data...")
    u = mda.Universe(args.structure, args.trajectory)

    print("Computing average contact map...")
    contact_avg = compute_contact_map(u, cutoff=args.cutoff, beta=args.beta)

    np.save(args.output, contact_avg)
    print(f"Average contact map saved to {args.output}")

if __name__ == "__main__":
    main()

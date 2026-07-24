import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.sparse.csgraph import connected_components
import argparse
import os

def tanh_switching(dist_matrix, d0, beta):
    """平滑 switching function，输出从1平滑过渡到0"""
    return 0.5 * (1 - np.tanh(beta * (dist_matrix - d0)))

def compute_chain_contact_matrix(bb_atoms, n_chains, atoms_per_chain, cutoff):
    """
    计算链间 contact 矩阵：
    若任意原子对距离 < cutoff，则两条链视为接触。
    返回布尔矩阵 contact_chains[i, j] = True / False
    """
    contact_chains = np.zeros((n_chains, n_chains), dtype=bool)
    for i in range(n_chains):
        start_i, end_i = i * atoms_per_chain, (i + 1) * atoms_per_chain
        pos_i = bb_atoms[start_i:end_i].positions
        for j in range(i + 1, n_chains):
            start_j, end_j = j * atoms_per_chain, (j + 1) * atoms_per_chain
            pos_j = bb_atoms[start_j:end_j].positions
            dmin = np.min(distance_array(pos_i, pos_j))
            if dmin < cutoff:
                contact_chains[i, j] = contact_chains[j, i] = True
    return contact_chains

def find_largest_cluster_from_contact(contact_matrix):
    """通过连通分量算法找到最大团簇（支持传递性定义）"""
    n_components, labels = connected_components(contact_matrix, directed=False)
    largest_label = np.bincount(labels).argmax()
    largest_indices = np.where(labels == largest_label)[0]
    return largest_indices

def compute_contact_map_largest_cluster(universe, selection='name BB', atoms_per_chain=140,
                                        cutoff=8.0, beta=1.0, contact_cutoff=10.0,
                                        fraction=0.2, save_cluster_info=None):
    """
    仅分析最后 fraction 部分的帧；
    每帧基于链间contact定义团簇，只分析最大团簇的平均 contact map；
    可输出每帧最大团簇链数信息。
    """
    bb_atoms = universe.select_atoms(selection)
    total_atoms = len(bb_atoms)
    n_chains = total_atoms // atoms_per_chain
    if total_atoms % atoms_per_chain != 0:
        raise ValueError("选中的原子总数不是每条链原子数的整数倍。")

    n_total_frames = len(universe.trajectory)
    start_frame = int((1 - fraction) * n_total_frames)

    contact_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
    n_frames = 0
    cluster_sizes = []

    print(f"检测到 {n_chains} 条链，每条链 {atoms_per_chain} 个原子。")
    print(f"仅分析最后 {fraction*100:.0f}% 帧 (从 {start_frame}/{n_total_frames} 开始)。")

    for iframe, ts in enumerate(universe.trajectory):
        if iframe < start_frame:
            continue  # 跳过前面的帧

        # Step 1: 找出链间 contact 矩阵
        contact_chains = compute_chain_contact_matrix(bb_atoms, n_chains, atoms_per_chain, contact_cutoff)

        # Step 2: 找最大团簇的链
        largest_cluster_indices = find_largest_cluster_from_contact(contact_chains)
        cluster_size = len(largest_cluster_indices)
        cluster_sizes.append((iframe, cluster_size))

        # Step 3: 对最大团簇中每条链计算 contact map 并平均
        frame_contact = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
        for i in largest_cluster_indices:
            start = i * atoms_per_chain
            end = (i + 1) * atoms_per_chain
            chain_atoms = bb_atoms[start:end]
            dist_matrix = distance_array(chain_atoms.positions, chain_atoms.positions)
            contact_map = tanh_switching(dist_matrix, d0=cutoff, beta=beta)
            np.fill_diagonal(contact_map, 1.0)
            frame_contact += contact_map

        frame_contact /= cluster_size
        contact_sum += frame_contact
        n_frames += 1

        if n_frames % 20 == 0:
            print(f"已处理 {n_frames} 帧 (frame {iframe})，最大团簇链数: {cluster_size}")

    contact_avg = contact_sum / n_frames

    print(f"分析完成，共 {n_frames} 帧。")

    # 如果用户要求保存团簇信息
    if save_cluster_info:
        np.savetxt(save_cluster_info, np.array(cluster_sizes, dtype=int),
                   fmt='%d', header='Frame\tLargestClusterSize')
        print(f"✅ 最大团簇链数信息已保存至 {save_cluster_info}")

    return contact_avg

def main():
    parser = argparse.ArgumentParser(description="Compute average contact map for largest cluster from trajectory.")
    parser.add_argument("-s", "--structure", required=True, help="Structure file (.tpr, .gro, .pdb)")
    parser.add_argument("-f", "--trajectory", required=True, help="Trajectory file (.xtc)")
    parser.add_argument("-o", "--output", default="contact_map_largest_cluster.npy", help="Output .npy file name")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Cutoff distance for contact map (Å)")
    parser.add_argument("--beta", type=float, default=1.0, help="Sharpness parameter for smooth switching function")
    parser.add_argument("--atoms_per_chain", type=int, default=140, help="Number of backbone atoms per chain")
    parser.add_argument("--contact_cutoff", type=float, default=10.0, help="Distance cutoff (Å) to define inter-chain contact")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction of last frames to analyze (e.g. 0.2 for last 20%)")
    parser.add_argument("--save_cluster_info", default=None, help="Optional output file to save largest cluster sizes per frame")
    args = parser.parse_args()

    if not os.path.exists(args.structure) or not os.path.exists(args.trajectory):
        raise FileNotFoundError("Structure or trajectory file not found.")

    print("加载轨迹中...")
    u = mda.Universe(args.structure, args.trajectory)

    print("开始计算最大团簇的平均 contact map...")
    contact_avg = compute_contact_map_largest_cluster(
        u,
        selection='name BB',
        atoms_per_chain=args.atoms_per_chain,
        cutoff=args.cutoff,
        beta=args.beta,
        contact_cutoff=args.contact_cutoff,
        fraction=args.fraction,
        save_cluster_info=args.save_cluster_info
    )

    np.save(args.output, contact_avg)
    print(f"✅ 平均 contact map 已保存至 {args.output}")

if __name__ == "__main__":
    main()

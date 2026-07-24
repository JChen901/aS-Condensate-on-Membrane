import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.sparse.csgraph import connected_components
import argparse
import os

def tanh_switching(dist_matrix, d0, beta):
    """平滑 switching function，输出从1平滑过渡到0"""
    return 0.5 * (1 - np.tanh(beta * (dist_matrix - d0)))

def compute_chain_contact_matrix(chain_coords, cutoff):
    """
    计算链间 contact 矩阵：
    若任意原子对距离 < cutoff，则两条链视为接触。
    返回布尔矩阵 contact_chains[i, j] = True / False
    """
    n_chains = chain_coords.shape[0]
    contact_chains = np.zeros((n_chains, n_chains), dtype=bool)
    for i in range(n_chains):
        pos_i = chain_coords[i]
        for j in range(i + 1, n_chains):
            pos_j = chain_coords[j]
            dmin = np.min(distance_array(pos_i, pos_j))
            if dmin < cutoff:
                contact_chains[i, j] = contact_chains[j, i] = True
    return contact_chains


def parse_chain_range(chain_range_str, n_chains):
    """
    解析 1-based 且包含端点的链范围字符串，例如 "61:100"。
    返回 0-based 索引数组。
    """
    try:
        start_str, end_str = chain_range_str.split(":")
        start = int(start_str)
        end = int(end_str)
    except Exception as exc:
        raise ValueError("--chain_range 格式应为 start:end，例如 61:100") from exc

    if start < 1 or end < 1:
        raise ValueError("--chain_range 使用 1-based 索引，start 和 end 必须 >= 1")
    if start > end:
        raise ValueError("--chain_range 要求 start <= end")
    if end > n_chains:
        raise ValueError(f"--chain_range 超出链总数 ({n_chains})")

    return np.arange(start - 1, end, dtype=int)

def find_largest_cluster_from_contact(contact_matrix):
    """通过连通分量算法找到最大团簇（支持传递性定义）"""
    n_components, labels = connected_components(contact_matrix, directed=False)
    largest_label = np.bincount(labels).argmax()
    largest_indices = np.where(labels == largest_label)[0]
    return largest_indices

def compute_contact_map_largest_cluster(universe, selection='name BB', atoms_per_chain=140,
                                        cutoff=8.0, beta=1.0, contact_cutoff=10.0,
                                        fraction=0.2, save_cluster_info=None,
                                        selected_chain_indices=None):
    """
    仅分析最后 fraction 部分的帧；
    每帧先基于全部链的链间 contact 判定全局最大团簇，
    然后仅对“最大团簇 ∩ 指定链范围”内的链做 intra-chain contact map 平均；
    可输出每帧全局最大团簇链数及交集链数信息。
    """
    bb_atoms = universe.select_atoms(selection)
    total_atoms = len(bb_atoms)
    n_chains = total_atoms // atoms_per_chain
    if total_atoms % atoms_per_chain != 0:
        raise ValueError("选中的原子总数不是每条链原子数的整数倍。")

    if selected_chain_indices is None:
        selected_chain_indices = np.arange(n_chains, dtype=int)
    else:
        selected_chain_indices = np.asarray(selected_chain_indices, dtype=int)
        if selected_chain_indices.ndim != 1 or selected_chain_indices.size == 0:
            raise ValueError("selected_chain_indices 必须是一维且非空。")
        if np.any(selected_chain_indices < 0) or np.any(selected_chain_indices >= n_chains):
            raise ValueError(f"selected_chain_indices 必须在 [0, {n_chains - 1}] 范围内。")
        selected_chain_indices = np.unique(selected_chain_indices)

    n_selected = len(selected_chain_indices)

    n_total_frames = len(universe.trajectory)
    start_frame = int((1 - fraction) * n_total_frames)

    contact_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
    n_frames = 0
    cluster_sizes = []

    print(f"检测到 {n_chains} 条链，每条链 {atoms_per_chain} 个原子。")
    print(f"指定链范围: {n_selected} (索引范围: {selected_chain_indices[0]} ~ {selected_chain_indices[-1]}, 0-based)")
    print(f"最大团簇判断基于全部 {n_chains} 条链，intra map 仅统计 '最大团簇 ∩ 指定范围'。")
    print(f"仅分析最后 {fraction*100:.0f}% 帧 (从 {start_frame}/{n_total_frames} 开始)。")

    for iframe, ts in enumerate(universe.trajectory):
        if iframe < start_frame:
            continue  # 跳过前面的帧

        # Step 1: 用全部链构建链间 contact 矩阵，判定全局最大团簇
        all_chain_coords = bb_atoms.positions.reshape(n_chains, atoms_per_chain, 3)
        contact_chains_all = compute_chain_contact_matrix(all_chain_coords, contact_cutoff)

        # Step 2: 全局最大团簇与指定链范围求交集
        largest_cluster_indices_global = find_largest_cluster_from_contact(contact_chains_all)
        intersection = np.intersect1d(largest_cluster_indices_global, selected_chain_indices)
        cluster_size_global = len(largest_cluster_indices_global)
        cluster_size_intersection = len(intersection)
        cluster_sizes.append((iframe, cluster_size_global, cluster_size_intersection))

        # Step 3: 对交集中的每条链计算 intra-chain contact map 并平均
        frame_contact = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
        if cluster_size_intersection > 0:
            for real_i in intersection:
                chain_pos = all_chain_coords[real_i]
                dist_matrix = distance_array(chain_pos, chain_pos)
                contact_map = tanh_switching(dist_matrix, d0=cutoff, beta=beta)
                np.fill_diagonal(contact_map, 1.0)
                frame_contact += contact_map

            frame_contact /= cluster_size_intersection
        contact_sum += frame_contact
        n_frames += 1

        if n_frames % 20 == 0:
            print(
                f"已处理 {n_frames} 帧 (frame {iframe})，"
                f"全局最大团簇链数: {cluster_size_global}，交集链数: {cluster_size_intersection}"
            )

    if n_frames == 0:
        contact_avg = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
    else:
        contact_avg = contact_sum / n_frames

    print(f"分析完成，共 {n_frames} 帧。")

    # 如果用户要求保存团簇信息
    if save_cluster_info:
        np.savetxt(save_cluster_info, np.array(cluster_sizes, dtype=int),
                   fmt='%d', header='Frame\tLargestClusterSizeGlobal\tIntersectionSize')
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
    parser.add_argument("--last_n_chains", type=int, default=None, help="Only use the last N chains for calculation (default: use all chains)")
    parser.add_argument("--chain_range", type=str, default=None, help="Use 1-based inclusive chain range, e.g. 61:100 (default: use all chains)")
    args = parser.parse_args()

    if not os.path.exists(args.structure) or not os.path.exists(args.trajectory):
        raise FileNotFoundError("Structure or trajectory file not found.")

    print("加载轨迹中...")
    u = mda.Universe(args.structure, args.trajectory)

    bb_atoms = u.select_atoms('name BB')
    total_atoms = len(bb_atoms)
    if total_atoms % args.atoms_per_chain != 0:
        raise ValueError("选中的原子总数不是每条链原子数的整数倍。")
    n_chains = total_atoms // args.atoms_per_chain

    if args.last_n_chains is not None and args.chain_range is not None:
        raise ValueError("--last_n_chains 和 --chain_range 不能同时使用。")

    selected_chain_indices = None
    if args.last_n_chains is not None:
        if args.last_n_chains <= 0:
            raise ValueError("--last_n_chains 必须 > 0")
        if args.last_n_chains > n_chains:
            raise ValueError(f"--last_n_chains 超过链总数 ({n_chains})")
        selected_chain_indices = np.arange(n_chains - args.last_n_chains, n_chains, dtype=int)
    elif args.chain_range is not None:
        selected_chain_indices = parse_chain_range(args.chain_range, n_chains)
    else:
        print("未指定 --last_n_chains 或 --chain_range，默认使用全部链进行计算。")

    print("开始计算最大团簇的平均 contact map...")
    contact_avg = compute_contact_map_largest_cluster(
        u,
        selection='name BB',
        atoms_per_chain=args.atoms_per_chain,
        cutoff=args.cutoff,
        beta=args.beta,
        contact_cutoff=args.contact_cutoff,
        fraction=args.fraction,
        save_cluster_info=args.save_cluster_info,
        selected_chain_indices=selected_chain_indices
    )

    np.save(args.output, contact_avg)
    print(f"✅ 平均 contact map 已保存至 {args.output}")

if __name__ == "__main__":
    main()

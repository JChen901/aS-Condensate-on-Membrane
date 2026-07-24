import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.sparse.csgraph import connected_components
import argparse
import os
import sys

def tanh_switching(dist_matrix, d0, beta):
    """平滑 switching function，输出从1平滑过渡到0"""
    return 0.5 * (1 - np.tanh(beta * (dist_matrix - d0)))

def compute_chain_contact_matrix(chain_coords, cutoff):
    """
    计算链间 contact 矩阵：
    返回布尔矩阵 contact_chains[i, j] = True (若两链最小距离 < cutoff)
    """
    n_chains = chain_coords.shape[0]
    contact_chains = np.zeros((n_chains, n_chains), dtype=bool)

    # 简单的双重循环 (可根据需要进一步优化，但基于原代码逻辑保持清晰)
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            dmin = np.min(distance_array(chain_coords[i], chain_coords[j]))
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
    """通过连通分量算法找到最大团簇的链索引"""
    n_components, labels = connected_components(contact_matrix, directed=False)
    largest_label = np.bincount(labels).argmax()
    largest_indices = np.where(labels == largest_label)[0]
    return largest_indices

def compute_inter_chain_contact_map_largest_cluster(universe, selection='name BB', atoms_per_chain=140,
                                                    cutoff=8.0, beta=1.0, contact_cutoff=10.0,
                                                    fraction=0.2, selected_chain_indices=None):
    """
    最大团簇基于全部链全局判断；contact map 仅计算"最大团簇 ∩ 指定链范围"内的链对。
    统计方式：每帧先对"该帧接触链对"做平均，再对帧做等权平均。
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

    # 累加矩阵（帧平均）
    frame_avg_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
    analyzed_frames = 0

    print(f"检测到 {n_chains} 条链，每条链 {atoms_per_chain} 个原子。")
    print(f"指定链范围: {n_selected} 条 (0-based 索引: {selected_chain_indices[0]} ~ {selected_chain_indices[-1]})")
    print(f"最大团簇判断基于全部 {n_chains} 条链，contact map 仅计算 '最大团簇 ∩ 指定范围' 内的链对。")
    print(f"仅分析最后 {fraction*100:.0f}% 帧 (从 {start_frame}/{n_total_frames} 开始)。")

    for iframe, ts in enumerate(universe.trajectory):
        if iframe < start_frame:
            continue

        analyzed_frames += 1

        all_chain_coords = bb_atoms.positions.reshape(n_chains, atoms_per_chain, 3)

        # Step 1: 使用全部链构建接触矩阵，全局判断最大团簇
        contact_chains_all = compute_chain_contact_matrix(all_chain_coords, contact_cutoff)
        largest_cluster_indices_global = find_largest_cluster_from_contact(contact_chains_all)

        # Step 2: 取最大团簇（全局0-based）与指定链范围的交集
        intersection = np.intersect1d(largest_cluster_indices_global, selected_chain_indices)

        # Step 3: 在交集内的链对上计算详细 Contact Map
        # 每帧先做"链对平均"；若交集为空或只有1条链，该帧记为全0矩阵。
        frame_contact_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
        frame_pairs_count = 0

        if len(intersection) >= 2:
            for idx_i in range(len(intersection)):
                real_i = intersection[idx_i]
                for idx_j in range(idx_i + 1, len(intersection)):
                    real_j = intersection[idx_j]

                    # 只有当这两条链确实直接接触时，才计算详细 Contact Map
                    if contact_chains_all[real_i, real_j]:
                        pos_i = all_chain_coords[real_i]
                        pos_j = all_chain_coords[real_j]

                        dist_matrix = distance_array(pos_i, pos_j)

                        # 计算平滑接触值 (0~1)，并显式对称化
                        cmap = tanh_switching(dist_matrix, d0=cutoff, beta=beta)
                        cmap_sym = 0.5 * (cmap + cmap.T)

                        frame_contact_sum += cmap_sym
                        frame_pairs_count += 1

        if frame_pairs_count > 0:
            frame_avg_map = frame_contact_sum / frame_pairs_count
        else:
            frame_avg_map = frame_contact_sum  # 全0

        frame_avg_sum += frame_avg_map

        if analyzed_frames % 10 == 0:
            sys.stdout.write(
                f"\r处理进度: Frame {iframe} | 全局最大团簇大小: {len(largest_cluster_indices_global)} | 交集链数: {len(intersection)} | 本帧接触对数: {frame_pairs_count}"
            )
            sys.stdout.flush()

    print(f"\n分析完成。")

    if analyzed_frames == 0:
        print("警告: 分析时间窗内没有可用帧。")
        return np.zeros((atoms_per_chain, atoms_per_chain))

    # 平均化：先帧内链对平均，再对帧等权平均
    contact_avg = frame_avg_sum / analyzed_frames
    
    return contact_avg

def main():
    parser = argparse.ArgumentParser(description="Compute average INTER-chain contact map for largest cluster.")
    parser.add_argument("-s", "--structure", required=True, help="Structure file (.tpr, .gro, .pdb)")
    parser.add_argument("-f", "--trajectory", required=True, help="Trajectory file (.xtc)")
    parser.add_argument("-o", "--output", default="inter_chain_contact_largest_cluster.npy", help="Output .npy file name")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Cutoff distance for contact map calculation (Å)")
    parser.add_argument("--beta", type=float, default=1.0, help="Sharpness parameter for switching function")
    parser.add_argument("--atoms_per_chain", type=int, default=140, help="Number of backbone atoms per chain")
    parser.add_argument("--contact_cutoff", type=float, default=10.0, help="Distance cutoff (Å) to define if two chains are in contact")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction of last frames to analyze")
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

    print("开始计算最大团簇内的链间 (Inter-chain) Contact Map...")
    contact_avg = compute_inter_chain_contact_map_largest_cluster(
        u,
        selection='name BB',
        atoms_per_chain=args.atoms_per_chain,
        cutoff=args.cutoff,
        beta=args.beta,
        contact_cutoff=args.contact_cutoff,
        fraction=args.fraction,
        selected_chain_indices=selected_chain_indices
    )

    np.save(args.output, contact_avg)
    print(f"✅ 平均链间 Contact Map 已保存至 {args.output}")

if __name__ == "__main__":
    main()
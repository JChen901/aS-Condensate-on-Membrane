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

def compute_chain_contact_matrix(bb_atoms, n_chains, atoms_per_chain, cutoff):
    """
    计算链间 contact 矩阵：
    返回布尔矩阵 contact_chains[i, j] = True (若两链最小距离 < cutoff)
    """
    contact_chains = np.zeros((n_chains, n_chains), dtype=bool)
    # 为了加速，提取所有链的坐标视图 (N_chains, N_atoms, 3)
    positions = bb_atoms.positions
    chain_coords = positions.reshape(n_chains, atoms_per_chain, 3)

    # 简单的双重循环 (可根据需要进一步优化，但基于原代码逻辑保持清晰)
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            dmin = np.min(distance_array(chain_coords[i], chain_coords[j]))
            if dmin < cutoff:
                contact_chains[i, j] = contact_chains[j, i] = True
    return contact_chains

def find_largest_cluster_from_contact(contact_matrix):
    """通过连通分量算法找到最大团簇的链索引"""
    n_components, labels = connected_components(contact_matrix, directed=False)
    largest_label = np.bincount(labels).argmax()
    largest_indices = np.where(labels == largest_label)[0]
    return largest_indices

def compute_inter_chain_contact_map_largest_cluster(universe, selection='name BB', atoms_per_chain=140,
                                                    cutoff=8.0, beta=1.0, contact_cutoff=10.0,
                                                    fraction=0.2):
    """
    只计算最大团簇内部，链与链之间的 Inter-chain Contact Map。
    统计方式：每帧先对“该帧接触链对”做平均，再对帧做等权平均。
    """
    bb_atoms = universe.select_atoms(selection)
    total_atoms = len(bb_atoms)
    n_chains = total_atoms // atoms_per_chain
    if total_atoms % atoms_per_chain != 0:
        raise ValueError("选中的原子总数不是每条链原子数的整数倍。")

    n_total_frames = len(universe.trajectory)
    start_frame = int((1 - fraction) * n_total_frames)

    # 累加矩阵（帧平均）
    frame_avg_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
    analyzed_frames = 0

    print(f"检测到 {n_chains} 条链，每条链 {atoms_per_chain} 个原子。")
    print(f"仅分析最后 {fraction*100:.0f}% 帧 (从 {start_frame}/{n_total_frames} 开始)。")

    for iframe, ts in enumerate(universe.trajectory):
        if iframe < start_frame:
            continue

        analyzed_frames += 1
        
        # Step 1: 计算链间接触拓扑 (定义谁和谁接触)
        # 这里直接用 contact_cutoff 来判断是否属于同一个团簇
        contact_chains = compute_chain_contact_matrix(bb_atoms, n_chains, atoms_per_chain, contact_cutoff)

        # Step 2: 找到最大团簇包含的链索引
        largest_cluster_indices = find_largest_cluster_from_contact(contact_chains)
        
        # Step 3: 遍历最大团簇内的链对
        # 每帧先做“链对平均”；若本帧没有接触链对，则该帧记为全0矩阵。
        frame_contact_sum = np.zeros((atoms_per_chain, atoms_per_chain), dtype=np.float32)
        frame_pairs_count = 0

        # 如果团簇只有1条链，本帧没有链间接触，保留全0 frame_map 参与帧平均
        if len(largest_cluster_indices) >= 2:
            cluster_positions = bb_atoms.positions.reshape(n_chains, atoms_per_chain, 3)
            indices = largest_cluster_indices

            for idx_i in range(len(indices)):
                real_i = indices[idx_i]
                for idx_j in range(idx_i + 1, len(indices)):
                    real_j = indices[idx_j]

                    # 只有当这两条链确实接触时，才计算详细 Contact Map
                    if contact_chains[real_i, real_j]:
                        pos_i = cluster_positions[real_i]
                        pos_j = cluster_positions[real_j]

                        dist_matrix = distance_array(pos_i, pos_j)

                        # 计算平滑接触值 (0~1)，并显式对称化
                        cmap = tanh_switching(dist_matrix, d0=cutoff, beta=beta)
                        cmap_sym = 0.5 * (cmap + cmap.T)

                        # 本帧内累加
                        frame_contact_sum += cmap_sym
                        frame_pairs_count += 1

        if frame_pairs_count > 0:
            frame_avg_map = frame_contact_sum / frame_pairs_count
        else:
            frame_avg_map = frame_contact_sum  # 全0

        frame_avg_sum += frame_avg_map

        if analyzed_frames % 10 == 0:
            sys.stdout.write(
                f"\r处理进度: Frame {iframe} | 本帧团簇大小: {len(largest_cluster_indices)} | 本帧接触对数: {frame_pairs_count}"
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
    
    args = parser.parse_args()

    if not os.path.exists(args.structure) or not os.path.exists(args.trajectory):
        raise FileNotFoundError("Structure or trajectory file not found.")

    print("加载轨迹中...")
    u = mda.Universe(args.structure, args.trajectory)

    print("开始计算最大团簇内的链间 (Inter-chain) Contact Map...")
    contact_avg = compute_inter_chain_contact_map_largest_cluster(
        u,
        selection='name BB',
        atoms_per_chain=args.atoms_per_chain,
        cutoff=args.cutoff,
        beta=args.beta,
        contact_cutoff=args.contact_cutoff,
        fraction=args.fraction
    )

    np.save(args.output, contact_avg)
    print(f"✅ 平均链间 Contact Map 已保存至 {args.output}")

if __name__ == "__main__":
    main()
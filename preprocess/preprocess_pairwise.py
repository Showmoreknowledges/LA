import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def load_edgelists_and_map_nodes(file_path1, file_path2):
    """
    加载两个edgelist文件，并构建一个覆盖两个图的全局节点映射。
    """
    all_nodes = set()
    edge_lists = []

    print(f"--- loading layers : '{file_path1}' 和 '{file_path2}' ---")
    for file_path in [file_path1, file_path2]:
        edges = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    if u == v: continue
                    edges.append((u, v))
                    all_nodes.add(u)
                    all_nodes.add(v)
        edge_lists.append(np.array(edges))

    # 创建从原始节点ID到0-N连续索引的映射
    node_map = {node_id: i for i, node_id in enumerate(sorted(list(all_nodes)))}
    
    return edge_lists[0], edge_lists[1], node_map

def main(args):
    edges1, edges2, node_map = load_edgelists_and_map_nodes(args.layer1, args.layer2)
    num_total_nodes = len(node_map)

    print(f"\n图加载完成: 两个图层共包含 {num_total_nodes} 个独立节点。")

    print("\n--- 构建两个图的边索引 ---")
    # 将原始节点ID转换为0-N的索引
    edge_index1 = np.array([[node_map[u], node_map[v]] for u, v in edges1]).T
    edge_index2 = np.array([[node_map[u], node_map[v]] for u, v in edges2]).T
    
    print(f"图1 ({args.layer1}): {edge_index1.shape[1]} 条边")
    print(f"图2 ({args.layer2}): {edge_index2.shape[1]} 条边")

    print("\n--- 生成对齐关系并划分为训练/验证/测试集 ---")
    # 所有节点都是完美对齐的：(i, i)
    all_known_pairs = np.array([[i, i] for i in range(num_total_nodes)])
    
    train_ratio, val_ratio, test_ratio = args.split_ratios
    
    # 第一次划分：分出训练集
    if train_ratio > 0:
        train_pairs, temp_pairs = train_test_split(
            all_known_pairs, train_size=train_ratio, random_state=args.seed
        )
    else:
        train_pairs, temp_pairs = np.array([]).reshape(0,2), all_known_pairs

    # 第二次划分：从剩余部分中分出验证集和测试集
    if val_ratio + test_ratio > 0 and len(temp_pairs) > 0:
        val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
        val_pairs, test_pairs = train_test_split(
            temp_pairs, train_size=val_ratio_in_temp, random_state=args.seed
        )
    else:
        val_pairs, test_pairs = np.array([]).reshape(0,2), np.array([]).reshape(0,2)


    print(f"对齐对划分完成: {len(train_pairs)} 训练, {len(val_pairs)} 验证, {len(test_pairs)} 测试。")

    print("\n--- 生成独热编码作为节点特征 ---")
    # 两个图共享同一套节点，所以特征矩阵是相同的单位矩阵
    features = np.eye(num_total_nodes, dtype=np.float32)

    print("\n--- 保存为 .npz 文件 ---")
    # 为了兼容我们现有的训练脚本，我们将训练对保存为'pos_pairs'，测试对为'test_pairs'
    # 我们的训练脚本会自动从'pos_pairs'中再分出一部分作为内部验证集
    np.savez(args.output_file, 
             x1=features, x2=features, 
             edge_index1=edge_index1, edge_index2=edge_index2,
             pos_pairs=np.vstack([train_pairs, val_pairs]), # 合并训练和验证对
             test_pairs=test_pairs)

    print(f"\n 预处理完成！文件已保存到: {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess two graph layers into a pairwise alignment .npz dataset.")
    parser.add_argument('--layer1', type=str, default=datasets/Twitter/, help='图层1的 .edgelist 文件路径')
    parser.add_argument('--layer2', type=str, default=datasets/Twitter/, help='图层2的 .edgelist 文件路径')
    parser.add_argument('--output_file', type=str, default=datasets, help='输出的 .npz 文件路径')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.2, 0.1, 0.7],
                            help='训练集、验证集、测试集的比例。示例: 0.2 0.1 0.7')
    parser.add_argument('--seed', type=int, default=42, help='用于数据划分的随机种子')
    
    args = parser.parse_args()
    main(args)
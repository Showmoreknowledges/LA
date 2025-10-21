# 包含：自动特征对齐、结构签名提取、拉普拉斯位置编码、拓扑中心性、簇标签提取
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

from torch_geometric.utils import degree, get_laplacian

# GPU 端结构特征计算（度中心性 + 拉普拉斯位置编码）
def calculate_graph_features_gpu(edge_index_tensor, num_nodes, lpe_dim):
    device = edge_index_tensor.device
    node_degrees = degree(edge_index_tensor[0], num_nodes=num_nodes).float()
    degree_centrality = (node_degrees / (num_nodes - 1)).unsqueeze(1)
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(
        edge_index_tensor, normalization='sym', num_nodes=num_nodes
    )
    L_sparse = torch.sparse_coo_tensor(laplacian_edge_index, laplacian_edge_weight, (num_nodes, num_nodes))
    eigvals, eigvecs = torch.linalg.eigh(L_sparse.to_dense())
    lpe_features = eigvecs[:, 1:lpe_dim+1]  # 忽略第一个（常数）特征向量
    return degree_centrality.cpu().numpy(), lpe_features.cpu().numpy()

# CPU端拓扑中心性计算（Katz + 介数）
def calculate_centrality_features_cpu(edge_index, num_nodes):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.T)
    betweenness = np.array([v for k, v in sorted(nx.betweenness_centrality(graph).items())]).reshape(-1, 1)
    try:
        katz_cent = nx.katz_centrality(graph, alpha=0.005, max_iter=5000)
    except nx.PowerIterationFailedConvergence:
        print("  - Katz中心性幂迭代失败，切换到更稳健的scipy求解器...")
        katz_cent = nx.katz_centrality(graph, alpha=0.005, solver='scipy')
    katz = np.array([v for k, v in sorted(katz_cent.items())]).reshape(-1, 1)
    return np.hstack([betweenness, katz])

# 对图节点进行聚类，返回簇标签（用于对齐前筛选）
def cluster_nodes(features, num_clusters):
    print(f"--- 对所有节点进行 KMeans 聚类 (K={num_clusters}) ---")
    km = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    cluster_labels = km.fit_predict(features)
    return cluster_labels

# 主数据预处理函数
def load_and_prepare_data(args):
    print(f"--- 正在从 '{args.data_path}' 加载数据 ---")
    data = np.load(args.data_path, allow_pickle=True)

    def safely_extract_array(data, key):
        if key not in data.keys(): return None
        array = data[key]; return array.item() if array.shape == () else array

    x1, x2 = safely_extract_array(data, 'x1'), safely_extract_array(data, 'x2')
    edge_index1, edge_index2 = safely_extract_array(data, 'edge_index1'), safely_extract_array(data, 'edge_index2')
    pos_pairs, test_pairs = safely_extract_array(data, 'pos_pairs'), safely_extract_array(data, 'test_pairs')

    # --- 特征缺省时的一热 ---
    if x1 is None:
        num_nodes_net1 = int(edge_index1.max() + 1)
        x1 = np.eye(num_nodes_net1, dtype=np.float32)
    if x2 is None:
        num_nodes_net2 = int(edge_index2.max() + 1)
        x2 = np.eye(num_nodes_net2, dtype=np.float32)

    # --- 统一特征维度 ---
    if x1.shape[1] != x2.shape[1]:
        print(f"警告: 两个图的原始特征维度不匹配 ({x1.shape[1]} vs {x2.shape[1]})，将进行零填充。")
        dim1, dim2 = x1.shape[1], x2.shape[1]
        if dim1 < dim2:
            x1 = np.hstack([x1, np.zeros((x1.shape[0], dim2 - dim1), dtype=np.float32)])
        else:
            x2 = np.hstack([x2, np.zeros((x2.shape[0], dim1 - dim2), dtype=np.float32)])
        print(f"  - 特征维度已统一为: {x1.shape[1]}")

    num_nodes_net1, num_nodes_net2 = x1.shape[0], x2.shape[0]
    semantic_features = np.vstack((x1, x2))
    topo_features, struct_features = None, None

    # --- 可选的图特征 ---
    if args.use_topo_features or args.use_struct_features:
        print("--- 正在计算额外的图特征 ---")
        edge_index1_t = torch.LongTensor(edge_index1).to(args.device)
        edge_index2_t = torch.LongTensor(edge_index2).to(args.device)

        degree1, struct1 = calculate_graph_features_gpu(edge_index1_t, num_nodes_net1, args.lpe_dim)
        degree2, struct2 = calculate_graph_features_gpu(edge_index2_t, num_nodes_net2, args.lpe_dim)

        if args.use_topo_features:
            print("  - 正在CPU上计算复杂的中心性...")
            centrality1 = calculate_centrality_features_cpu(edge_index1, num_nodes_net1)
            centrality2 = calculate_centrality_features_cpu(edge_index2, num_nodes_net2)
            topo1 = np.hstack([degree1, centrality1])
            topo2 = np.hstack([degree2, centrality2])
            topo_features = np.vstack((topo1, topo2))
            print(f"  - 拓扑特征已计算，维度: {topo_features.shape[1]}")

        if args.use_struct_features:
            struct_features = np.vstack((struct1, struct2))
            print(f"  - 结构特征 (LPE) 已在GPU上计算，维度: {struct_features.shape[1]}")

    # --- 可选 PCA 降维 ---
    if args.use_pca and semantic_features.shape[1] > args.pca_dim:
        print(f"--- 正在对语义特征进行PCA降维 (从 {semantic_features.shape[1]}维 -> {args.pca_dim}维) ---")
        pca = PCA(n_components=args.pca_dim)
        semantic_features = pca.fit_transform(semantic_features)

    feature_list = [semantic_features]
    if topo_features is not None: feature_list.append(topo_features)
    if struct_features is not None: feature_list.append(struct_features)
    features = np.hstack(feature_list)

    semantic_dim = semantic_features.shape[1]
    topo_dim = topo_features.shape[1] if topo_features is not None else 0
    struct_dim = struct_features.shape[1] if struct_features is not None else 0

    edge_index = np.hstack((edge_index1, edge_index2))

    # --- 关键修改：训练时划分，兼容两种 npz 结构 ---
    # 若 test_pairs 不存在，则直接使用 pos_pairs 作为“全部已知对齐对”
    if pos_pairs is None and test_pairs is None:
        raise ValueError("数据文件中既没有 pos_pairs 也没有 test_pairs，无法进行对齐训练。")

    if test_pairs is None or (isinstance(test_pairs, np.ndarray) and test_pairs.size == 0):
        all_known_pairs = pos_pairs
    else:
        # 兼容老数据：若两者都存在则合并
        if pos_pairs.ndim == 1: pos_pairs = pos_pairs.reshape(-1, 2)
        if test_pairs.ndim == 1: test_pairs = test_pairs.reshape(-1, 2)
        all_known_pairs = np.vstack((pos_pairs, test_pairs))

    # --- 可选的聚类筛选在“全体对齐对”上进行 ---
    working_pairs = all_known_pairs
    if getattr(args, 'cluster_align', False):
        print("--- 启用聚类对齐筛选策略 ---")
        cluster_labels = cluster_nodes(features, num_clusters=args.cluster_k)
        cluster1, cluster2 = cluster_labels[:num_nodes_net1], cluster_labels[num_nodes_net1:]
        filtered = []
        for i, j in working_pairs:
            if cluster1[i] == cluster2[j]:
                filtered.append([i, j])
        print(f"  - 聚类筛选后，剩余对数: {len(filtered)} / {len(working_pairs)}")
        working_pairs = np.array(filtered, dtype=np.int64)

    # --- 按传入比例进行训练期划分 ---
    train_ratio, val_ratio, test_ratio = args.split_ratios
    from sklearn.model_selection import train_test_split

    # 首先划出训练
    train_pairs, temp_pairs = train_test_split(
        working_pairs, train_size=train_ratio, random_state=args.seed, shuffle=True
    )
    # 再把剩余划成 val/test
    if len(temp_pairs) > 0 and (val_ratio + test_ratio) > 0:
        val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
        val_pairs, test_pairs = train_test_split(
            temp_pairs, train_size=val_ratio_in_temp, random_state=args.seed, shuffle=True
        )
    else:
        val_pairs, test_pairs = np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    # --- 打包 Tensor ---
    features_tensor = torch.FloatTensor(features)
    edge_index_tensor = torch.LongTensor(edge_index)
    train_pairs_tensor = torch.LongTensor(train_pairs)
    val_pairs_tensor = torch.LongTensor(val_pairs)
    test_pairs_tensor = torch.LongTensor(test_pairs)
    batch_mapping = torch.cat([
    torch.zeros(num_nodes_net1, dtype=torch.long),
    torch.ones(num_nodes_net2, dtype=torch.long)
    ])

    print("数据准备完成:")
    print(f"  - 总节点数: {features_tensor.shape[0]}")
    print(f"  - 最终特征维度: {features.shape[1]} (语义:{semantic_dim}, 拓扑:{topo_dim}, 结构:{struct_dim})")
    print(f"  - 训练对数量: {train_pairs_tensor.shape[0]} ({train_ratio:.0%})")
    print(f"  - 验证对数量: {val_pairs_tensor.shape[0]} ({val_ratio:.0%})")
    print(f"  - 测试对数量: {test_pairs_tensor.shape[0]} ({test_ratio:.0%})")

    return {
        'features': features_tensor.to(args.device),
        'edge_index': edge_index_tensor.to(args.device),
        'batch_mapping': batch_mapping.to(args.device),
        'train_pairs': train_pairs_tensor.to(args.device),
        'val_pairs': val_pairs_tensor.to(args.device),
        'test_pairs': test_pairs_tensor.to(args.device),
        'num_nodes_net1': num_nodes_net1, 'num_nodes_net2': num_nodes_net2,
        'semantic_dim': semantic_dim, 'topo_dim': topo_dim, 'struct_dim': struct_dim
    }
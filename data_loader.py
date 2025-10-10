import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # 用于降维

def load_and_prepare_data(args):
    """
    加载.npz文件，智能处理不同格式（包括特征缺失），并划分为训练、验证、测试三部分。
    """
    print(f"--- 正在从 '{args.data_path}' 加载数据 ---")
    
    try:
        data = np.load(args.data_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{args.data_path}' 未找到。")
        exit()

    def safely_extract_array(data, key):
        if key not in data.keys():
            return None
        array = data[key]
        if array.shape == ():
            return array.item()
        return array

    x1 = safely_extract_array(data, 'x1')
    x2 = safely_extract_array(data, 'x2')
    edge_index1 = safely_extract_array(data, 'edge_index1')
    edge_index2 = safely_extract_array(data, 'edge_index2')
    pos_pairs = safely_extract_array(data, 'pos_pairs')
    test_pairs = safely_extract_array(data, 'test_pairs')
    
    # 处理节点特征缺失的情况
    if x1 is None:
        if edge_index1 is not None and edge_index1.size > 0:
            num_nodes_net1 = int(edge_index1.max() + 1)
            print(f"警告: 图1的节点特征 ('x1') 未找到。将自动为 {num_nodes_net1} 个节点创建独热编码特征。")
            x1 = np.eye(num_nodes_net1, dtype=np.float32)
        else:
            raise ValueError("无法确定图1的节点数量：'x1' 和 'edge_index1' 均缺失或为空。")

    if x2 is None:
        if edge_index2 is not None and edge_index2.size > 0:
            num_nodes_net2 = int(edge_index2.max() + 1)
            print(f"警告: 图2的节点特征 ('x2') 未找到。将自动为 {num_nodes_net2} 个节点创建独热编码特征。")
            x2 = np.eye(num_nodes_net2, dtype=np.float32)
        else:
            raise ValueError("无法确定图2的节点数量：'x2' 和 'edge_index2' 均缺失或为空。")

# --- 根据命令行参数执行PCA降维 (已修复缩进) ---
    if args.use_pca and x1 is not None and x1.shape[1] > args.pca_dim:
        print(f"--- 正在对输入特征进行PCA降维 (从 {x1.shape[1]}维 -> {args.pca_dim}维) ---")

        # 将两个特征矩阵合并进行PCA，保证变换空间的一致性
        combined_x = np.vstack((x1, x2))
        pca = PCA(n_components=args.pca_dim)
        transformed_x = pca.fit_transform(combined_x)

        # 重新切分回 x1 和 x2
        x1 = transformed_x[:x1.shape[0], :]
        x2 = transformed_x[x1.shape[0]:, :]
        print(f"  - PCA降维完成。新特征维度: {x1.shape[1]}")

    # --- 核心:在堆叠前，对齐特征维度 ---
    if x1.shape[1] != x2.shape[1]:
        print("警告: 两个图的特征维度不匹配。将对维度较小的特征矩阵进行零填充。")
        dim1, dim2 = x1.shape[1], x2.shape[1]
        if dim1 < dim2:
            # 填充 x1
            padding = np.zeros((x1.shape[0], dim2 - dim1), dtype=np.float32)
            x1 = np.hstack([x1, padding])
        else:
            # 填充 x2
            padding = np.zeros((x2.shape[0], dim1 - dim2), dtype=np.float32)
            x2 = np.hstack([x2, padding])
        print(f"  - 特征维度已统一为: {x1.shape[1]}")
    
    # 合并 pos_pairs 和 test_pairs
    all_known_pairs = np.vstack((pos_pairs, test_pairs))
    print(f"发现总共 {len(all_known_pairs)} 个已知对齐对。")

    num_nodes_net1 = x1.shape[0]
    features = np.vstack((x1, x2))
    edge_index2_adjusted = edge_index2 + num_nodes_net1
    edge_index = np.hstack((edge_index1, edge_index2_adjusted))

    # 根据传入的比例进行两次划分
    train_ratio, val_ratio, test_ratio = args.split_ratios
    
    train_pairs, temp_pairs = train_test_split(
        all_known_pairs,
        train_size=train_ratio,
        random_state=args.seed
    )
    
    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 1.0
    if len(temp_pairs) > 0:
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            train_size=val_ratio_in_temp,
            random_state=args.seed
        )
    else:
        val_pairs, test_pairs = np.array([]).reshape(0,2), np.array([]).reshape(0,2)

    # 转换为PyTorch张量
    features_tensor = torch.FloatTensor(features)
    edge_index_tensor = torch.LongTensor(edge_index)
    train_pairs_tensor = torch.LongTensor(train_pairs)
    val_pairs_tensor = torch.LongTensor(val_pairs)
    test_pairs_tensor = torch.LongTensor(test_pairs)
    
    batch_mapping = torch.zeros(features_tensor.shape[0], dtype=torch.long)

    print("数据准备完成:")
    print(f"  - node num: {features_tensor.shape[0]}")
    print(f"  - 训练对数量: {train_pairs_tensor.shape[0]} ({train_ratio:.0%})")
    print(f"  - 验证对数量: {val_pairs_tensor.shape[0]} ({val_ratio:.0%})")
    print(f"  - 测试对数量: {test_pairs_tensor.shape[0]} ({test_ratio:.0%})")
    
    data_dict = {
        'features': features_tensor.to(args.device),
        'edge_index': edge_index_tensor.to(args.device),
        'batch_mapping': batch_mapping.to(args.device),
        'train_pairs': train_pairs_tensor.to(args.device),
        'val_pairs': val_pairs_tensor.to(args.device),
        'test_pairs': test_pairs_tensor.to(args.device),
        'num_nodes_net1': num_nodes_net1,
        'num_nodes_net2': x2.shape[0]
    }
    
    return data_dict
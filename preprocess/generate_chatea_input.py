import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# 确保可以从我们自己的项目中导入模型
from model import NodeEmbeddingModel

def get_args():
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate Input Files for ChatEA from ESA Embeddings")
    
    # --- 输入数据 ---
    parser.add_argument('--data_path', type=str, default='ACM-DBLP_0.2.npz',
                        help='输入数据文件的路径 (.npz格式)')
    
    # --- ESA 模型配置 (应与训练/使用的模型一致) ---
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 128])
    parser.add_argument('--num_heads', type=int, nargs='+', default=[4, 4, 8])
    parser.add_argument('--layer_types', type=str, nargs='+', default=['M', 'M', 'S'])
    
    # --- ChatEA 输出配置 ---
    parser.add_argument('--output_dir', type=str, default='data/ACM-DBLP/candidates',
                        help='ChatEA输入文件的输出目录')
    parser.add_argument('--top_k', type=int, default=20,
                        help='为每个实体生成的候选数量')
    
    # --- 运行配置 ---
    parser.add_argument('--device', type=str, default='auto',
                           help="运行设备 ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def get_node_embeddings(args, features_tensor, edge_index_tensor, batch_mapping):
    """使用ESA模型生成节点嵌入。"""
    print("\n--- 正在初始化并运行 ESA 模型以生成节点嵌入 ---")
    model = NodeEmbeddingModel(
        in_features=features_tensor.shape[1],
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        num_heads=args.num_heads,
        layer_types=args.layer_types,
        use_mlp=True, # 假设总是使用MLP
        dropout=0.1
    ).to(args.device)

    model.eval()
    with torch.no_grad():
        node_embeddings = model(features_tensor, edge_index_tensor, batch_mapping)
    
    print(f"✅ 节点嵌入生成成功！形状: {node_embeddings.shape}")
    return node_embeddings.cpu().numpy()

def main(args):
    # 1. 加载原始数据
    print(f"--- 正在从 '{args.data_path}' 加载原始数据 ---")
    data = np.load(args.data_path)
    x1, x2 = data['x1'], data['x2']
    edge_index1, edge_index2 = data['edge_index1'], data['edge_index2']
    # ChatEA的评估通常在测试集上进行
    alignment_pairs_to_process = data['test_pairs'] 

    num_nodes_net1 = x1.shape[0]
    
    # 2. 生成节点嵌入
    features = np.vstack((x1, x2))
    edge_index2_adjusted = edge_index2 + num_nodes_net1
    edge_index = np.hstack((edge_index1, edge_index2_adjusted))
    
    features_tensor = torch.FloatTensor(features).to(args.device)
    edge_index_tensor = torch.LongTensor(edge_index).to(args.device)
    batch_mapping = torch.zeros(features.shape[0], dtype=torch.long).to(args.device)
    
    embeddings = get_node_embeddings(args, features_tensor, edge_index_tensor, batch_mapping)
    
    embeds1 = embeddings[:num_nodes_net1]
    embeds2 = embeddings[num_nodes_net1:]

    # 3. 计算相似度并生成候选集
    print(f"\n--- 正在为 {len(alignment_pairs_to_process)} 个实体生成 Top-{args.top_k} 候选集 ---")
    source_embeds = embeds1[alignment_pairs_to_process[:, 0]]
    sim_matrix = cosine_similarity(source_embeds, embeds2)
    
    # 4. 构建 ChatEA 所需的数据结构
    chatea_cand = {}
    chatea_neighbors = {}
    chatea_name_dict = {'ent': {}, 'rel': {0: 'related_to'}} # 假设只有一种关系

    for i in range(features.shape[0]):
        chatea_name_dict['ent'][i] = f"Entity_{i}"

    all_edge_index = np.hstack((edge_index1, edge_index2_adjusted))
    for i in range(all_edge_index.shape[1]):
        h, t = all_edge_index[0, i], all_edge_index[1, i]
        if h not in chatea_neighbors: chatea_neighbors[h] = []
        if t not in chatea_neighbors: chatea_neighbors[t] = []
        chatea_neighbors[h].append((str(h), str(0), str(t)))
        chatea_neighbors[t].append((str(t), str(0), str(h)))

    print("--- 正在格式化为 ChatEA 输入文件 ---")
    for i, (source_id, ref_id) in enumerate(tqdm(alignment_pairs_to_process, desc="Processing entities")):
        sim_vector = sim_matrix[i, :]
        top_k_indices_in_net2 = np.argsort(-sim_vector)[:args.top_k]
        top_k_scores = sim_vector[top_k_indices_in_net2]
        
        ground_rank_list = np.where(top_k_indices_in_net2 == ref_id)[0]
        ground_rank = ground_rank_list[0] if len(ground_rank_list) > 0 else -1

        # ChatEA 的ID是全局的，所以需要将网络2的ID加上网络1的节点数
        chatea_cand[str(source_id)] = {
            'ref': int(ref_id + num_nodes_net1),
            'ground_rank': int(ground_rank),
            'candidates': [int(idx + num_nodes_net1) for idx in top_k_indices_in_net2],
            'cand_sims': [float(s) for s in top_k_scores]
        }
        
    # 5. 保存文件
    print(f"\n--- 正在保存文件到 '{args.output_dir}' ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'cand'), 'w') as f:
        json.dump(chatea_cand, f)
    print("  - 'cand' 文件已保存。")

    with open(os.path.join(args.output_dir, 'neighbors'), 'w') as f:
        json.dump(chatea_neighbors, f)
    print("  - 'neighbors' 文件已保存。")
    
    with open(os.path.join(args.output_dir, 'name_dict'), 'w') as f:
        json.dump(chatea_name_dict, f)
    print("  - 'name_dict' 文件已保存。")
    
    print("\n✅ ChatEA 输入文件生成完毕！现在可以运行 ChatEA。")


if __name__ == '__main__':
    args = get_args()
    # 复用我们之前的 config.py 来管理参数
    from config import get_args as get_esa_args
    esa_args = get_esa_args()
    # 将 ESA 的参数合并到当前脚本的参数中
    vars(args).update(vars(esa_args))
    main(args)
import torch
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 本地模块导入
from data_loader import load_and_prepare_data
from model import NodeEmbeddingModel

def get_args():
    parser = argparse.ArgumentParser(description="ESA-based Node Embedding Model Training")

    # --- 数据相关参数 ---
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_path', type=str, default='datasets/ACM-DBLP_0.2.npz',help='输入数据文件的路径 (.npz格式)')
    data_group.add_argument('--split_ratios', type=float, nargs=3, default=[0.15, 0.05, 0.8],
                            help='训练集、验证集、测试集的比例。三者之和应为1.0。示例: 0.15 0.05 0.8')
    data_group.add_argument('--seed', type=int, default=42, help='用于数据划分和初始化的随机种子')

    # --- 模型架构参数 ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embed_dim', type=int, default=128,help='最终输出的节点嵌入维度')
    model_group.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 128],help='每个注意力层的隐藏维度列表')
    model_group.add_argument('--num_heads', type=int, nargs='+', default=[4, 4, 8],help='每个注意力层的头数列表')
    model_group.add_argument('--layer_types', type=str, nargs='+', default=['M', 'M', 'S'],
                             choices=['M', 'S'],
                             help="注意力层的类型序列 ('M' for Masked, 'S' for Standard)")
    model_group.add_argument('--no_mlp', action='store_true',
                             help='如果设置，则不在注意力块后使用MLP')
    model_group.add_argument('--dropout', type=float, default=0.1,
                             help='模型中使用的dropout率')

    # --- 训练过程参数 ---
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--lr', type=float, default=0.001, help='学习率 (learning rate)')
    train_group.add_argument('--weight_decay', type=float, default=1e-5, 
                             help='权重衰减 (weight decay)')
    train_group.add_argument('--epochs', type=int, default=500, help='训练轮数 (epochs)')
    train_group.add_argument('--eval_interval', type=int, default=10, 
                             help='每隔多少轮在验证集上评估一次')
    train_group.add_argument('--margin', type=float, default=2.0, 
                             help='损失函数中的间隔')
    train_group.add_argument('--neg_sample_k', type=int, default=10, 
                             help='困难负采样中的k值')
    
    # --- 运行设备参数 ---
    run_group = parser.add_argument_group('Runtime Parameters')
    run_group.add_argument('--device', type=str, default='auto',
                           help="device ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()

    # --- 参数校验与后处理 ---
    if not np.isclose(sum(args.split_ratios), 1.0):
        raise ValueError(f"split_ratios 的总和必须为 1.0，当前为 {sum(args.split_ratios)}")
    if len(args.hidden_dims) != len(args.num_heads) or len(args.hidden_dims) != len(args.layer_types):
        raise ValueError("`hidden_dims`, `num_heads`, 和 `layer_types` 的列表长度必须一致。")
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args

def get_hard_negative_pairs(pos_pairs, embeds1, embeds2, k):
    source_embeds = embeds1[pos_pairs[:, 0]]
    sim_matrix = torch.matmul(source_embeds, embeds2.T)
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    neg_targets = []
    for i in range(pos_pairs.shape[0]):
        true_target_id = pos_pairs[i, 1]
        possible_negatives = top_k_indices[i]
        hard_neg_candidates = possible_negatives[possible_negatives != true_target_id]
        if len(hard_neg_candidates) > 0:
            chosen_neg_id = hard_neg_candidates[torch.randint(0, len(hard_neg_candidates), (1,))]
            neg_targets.append(chosen_neg_id)
        else:
            rand_neg_id = torch.randint(0, embeds2.shape[0], (1,), device=pos_pairs.device)
            neg_targets.append(rand_neg_id)
    neg_targets = torch.cat(neg_targets, dim=0)
    return torch.stack((pos_pairs[:, 0], neg_targets), dim=1)

@torch.no_grad()
def evaluate(model, data, split='val'):
    """在指定的数据集（验证集或测试集）上评估模型"""
    model.eval()
    
    all_embeddings = model(data['features'], data['edge_index'], data['batch_mapping'])
    embeds1 = all_embeddings[:data['num_nodes_net1']]
    embeds2 = all_embeddings[data['num_nodes_net1']:]

    if split == 'val':
        pairs_to_eval = data['val_pairs']
    elif split == 'test':
        pairs_to_eval = data['test_pairs']
    else:
        raise ValueError("split 参数必须是 'val' 或 'test'")

    source_embeds = embeds1[pairs_to_eval[:, 0]]
    sim_matrix = cosine_similarity(source_embeds.cpu().numpy(), embeds2.cpu().numpy())

    ranks = []
    for i in range(len(pairs_to_eval)):
        true_target_id = pairs_to_eval[i, 1].item()
        scores = sim_matrix[i, :]
        sorted_indices = np.argsort(-scores)
        rank = np.where(sorted_indices == true_target_id)[0][0]
        ranks.append(rank + 1)
        
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks <= 1)
    hits_at_10 = np.mean(ranks <= 10)
    
    return mrr, hits_at_1, hits_at_10


def main(args):
    data = load_and_prepare_data(args)

    model = NodeEmbeddingModel(
        in_features=data['features'].shape[1],
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        num_heads=args.num_heads,
        layer_types=args.layer_types,
        use_mlp=not args.no_mlp,
        dropout=args.dropout
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)

    best_mrr = 0.0
    best_model_state = None

    print("\n--- 开始训练 ---")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        all_embeddings = model(data['features'], data['edge_index'], data['batch_mapping'])
        embeds1 = all_embeddings[:data['num_nodes_net1']]
        embeds2 = all_embeddings[data['num_nodes_net1']:]

        pos_pairs = data['train_pairs']
        neg_pairs = get_hard_negative_pairs(pos_pairs, embeds1.detach(), embeds2.detach(), k=args.neg_sample_k)
        
        pos_src_embeds = embeds1[pos_pairs[:, 0]]
        pos_tgt_embeds = embeds2[pos_pairs[:, 1]]
        neg_src_embeds = embeds1[neg_pairs[:, 0]]
        neg_tgt_embeds = embeds2[neg_pairs[:, 1]]

        pos_dist = torch.sum((pos_src_embeds - pos_tgt_embeds)**2, dim=1)
        neg_dist = torch.sum((neg_src_embeds - neg_tgt_embeds)**2, dim=1)
        target = -torch.ones(pos_dist.shape[0], device=args.device)
        loss = loss_fn(pos_dist, neg_dist, target)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            mrr, h1, h10 = evaluate(model, data, split='val')
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val MRR: {mrr:.4f} | Val H@1: {h1:.4f} | Val H@10: {h10:.4f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                best_model_state = model.state_dict()
                print(f"🚀 New best model found at epoch {epoch+1} with MRR: {best_mrr:.4f}")

    print("\n训练完成！")

    if best_model_state:
        print("\n--- 加载最佳模型并在测试集上进行最终评估 ---")
        model.load_state_dict(best_model_state)
        test_mrr, test_h1, test_h10 = evaluate(model, data, split='test')
        print("\n--- The final result ---")
        print(f"  - Test MRR:     {test_mrr:.4f}")
        print(f"  - Test Hits@1:  {test_h1:.4f}")
        print(f"  - Test Hits@10: {test_h10:.4f}")
    else:
        print("\n未能找到最佳模型，跳过最终测试。")

if __name__ == '__main__':
    args = get_args()
    main(args)
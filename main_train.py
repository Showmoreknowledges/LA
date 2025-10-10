import torch
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# 本地模块导入
from data_loader import load_and_prepare_data
from model import NodeEmbeddingModel

def get_args():
    parser = argparse.ArgumentParser(description="Advanced ESA Model Training with InfoNCE Loss")

    # --- 数据相关参数 ---
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_path', type=str, default='datasets/Douban_0.2.npz',help='输入数据文件的路径 (.npz格式)')
    data_group.add_argument('--split_ratios', type=float, nargs=3, default=[0.15, 0.05, 0.8],
                            help='训练集、验证集、测试集的比例。')
    data_group.add_argument('--seed', type=int, default=42, help='用于数据划分和初始化的随机种子')

    # --- 模型架构参数 ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embed_dim', type=int, default=128,help='最终输出的节点嵌入维度')
    model_group.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 128],help='每个注意力层的隐藏维度列表')
    model_group.add_argument('--num_heads', type=int, nargs='+', default=[4, 4, 8],help='每个注意力层的头数列表')
    model_group.add_argument('--layer_types', type=str, nargs='+', default=['M', 'M', 'S'],
                             choices=['M', 'S'],
                             help="注意力层的类型序列 ('M' for Masked, 'S' for Standard)")
    model_group.add_argument('--no_mlp', action='store_true',help='如果设置，则不在注意力块后使用MLP')
    model_group.add_argument('--dropout', type=float, default=0.1,help='模型中使用的dropout率')
    model_group.add_argument('--use_pca', action='store_true' ,help='是否对输入特征使用PCA降维')
    model_group.add_argument('--pca_dim', type=int, default=128,help='PCA降维后的目标维度')

    # --- 训练过程参数 ---
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--lr', type=float, default=0.0005, help='学习率')
    train_group.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    train_group.add_argument('--epochs', type=int, default=500, help='训练轮数')
    train_group.add_argument('--eval_interval', type=int, default=10, help='评估间隔')
    train_group.add_argument('--neg_sample_k', type=int, default=128, # InfoNCE 通常受益于更多的负样本
                             help='每个正样本对使用的负样本数量')
    train_group.add_argument('--temperature', type=float, default=0.1,
                             help='InfoNCE loss 中的温度系数')

    # --- 运行设备参数 ---
    run_group = parser.add_argument_group('Runtime Parameters')
    run_group.add_argument('--device', type=str, default='auto',help="device ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()
    
    # 后处理与校验
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 其他校验...
    return args

def get_negative_samples(pos_pairs, num_nodes_net2, k, device):
    """
    为每个正样本对随机采样k个负样本。
    InfoNCE不一定需要困难负采样，随机采样通常效果已经很好，且效率更高。
    """
    pos_src = pos_pairs[:, 0].unsqueeze(1) # [N, 1]
    neg_candidates = torch.randint(0, num_nodes_net2, (pos_pairs.shape[0], k), device=device) # [N, K]
    
    # 简单地将源节点与随机负样本组合
    neg_pairs_src = pos_src.expand(-1, k).flatten() # [N*K]
    neg_pairs_tgt = neg_candidates.flatten() # [N*K]
    
    return neg_pairs_src, neg_pairs_tgt


def info_nce_loss(pos_sim, neg_sim, temperature):
    """
    计算 InfoNCE Loss。
    pos_sim: [N], 正样本对的相似度
    neg_sim: [N, K], N个源节点分别与K个负样本的相似度
    """
    # 将正样本相似度也加入logits中
    # logits 形状: [N, 1+K]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    logits /= temperature
    
    # 标签是第一列 (正样本)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(model, data, split='val'):
    # 评估函数保持不变
    model.eval()
    all_embeddings = model(data['features'], data['edge_index'], data['batch_mapping'])
    embeds1 = all_embeddings[:data['num_nodes_net1']]
    embeds2 = all_embeddings[data['num_nodes_net1']:]

    pairs_to_eval = data[f'{split}_pairs']
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
    mrr, h1, h10 = np.mean(1.0 / ranks), np.mean(ranks <= 1), np.mean(ranks <= 10)
    return mrr, h1, h10

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
    
    best_mrr = 0.0
    best_model_state = None

    print("\n--- 开始训练 (InfoNCE Loss) ---")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        all_embeddings = model(data['features'], data['edge_index'], data['batch_mapping'])
        embeds1 = all_embeddings[:data['num_nodes_net1']]
        embeds2 = all_embeddings[data['num_nodes_net1']:]

        pos_pairs = data['train_pairs']
        
        # --- 损失计算流程修改 ---
        pos_src_embeds = embeds1[pos_pairs[:, 0]]
        pos_tgt_embeds = embeds2[pos_pairs[:, 1]]
        
        # 1. 计算正样本对的余弦相似度
        pos_sim = torch.sum(F.normalize(pos_src_embeds) * F.normalize(pos_tgt_embeds), dim=1)

        # 2. 获取负样本
        neg_src_idx, neg_tgt_idx = get_negative_samples(pos_pairs, data['num_nodes_net2'], args.neg_sample_k, args.device)
        neg_src_embeds = embeds1[neg_src_idx]
        neg_tgt_embeds = embeds2[neg_tgt_idx]

        # 3. 计算负样本对的余弦相似度
        neg_sim_flat = torch.sum(F.normalize(neg_src_embeds) * F.normalize(neg_tgt_embeds), dim=1)
        # 将其重塑为 [N, K] 的形状，以便每个正样本都有K个对应的负样本相似度
        neg_sim = neg_sim_flat.view(pos_pairs.shape[0], args.neg_sample_k)
        
        # 4. 计算 InfoNCE Loss
        loss = info_nce_loss(pos_sim, neg_sim, args.temperature)

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
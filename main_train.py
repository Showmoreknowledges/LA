import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

from data_loader import load_and_prepare_data
from model import NodeEmbeddingModel


def get_args():
    parser = argparse.ArgumentParser(description="Advanced ESA Model Training with Multi-Feature Fusion")
    # --- 数据相关参数 ---
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_path', type=str, default='datasets/Douban_0.2.npz',help='输入数据文件的路径 (.npz格式)')
    data_group.add_argument('--split_ratios', type=float, nargs=3, default=[0.15, 0.05, 0.8], help='训练、验证、测试集的比例。')
    data_group.add_argument('--seed', type=int, default=42, help='用于数据划分和初始化的随机种子')
    data_group.add_argument('--use_pca', action='store_true', help='【新增】如果设置，则对高维输入特征使用PCA降维')
    data_group.add_argument('--pca_dim', type=int, default=128, help='【新增】PCA降维后的目标维度')
    data_group.add_argument('--use_topo_features', action='store_true', help='【新增】计算并使用拓扑特征(度,介数,Katz)')
    data_group.add_argument('--use_struct_features', action='store_true', help='【新增】计算并使用结构特征(LPE)')
    data_group.add_argument('--lpe_dim', type=int, default=16, help='【新增】拉普拉斯位置编码(LPE)的目标维度')

    # --- 模型架构参数 ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embed_dim', type=int, default=128,help='最终输出的节点嵌入维度')
    model_group.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128],help='每个注意力层的隐藏维度列表')
    model_group.add_argument('--num_heads', type=int, nargs='+', default=[4, 8],help='每个注意力层的头数列表')
    model_group.add_argument('--layer_types', type=str, nargs='+', default=['M', 'S'], choices=['M', 'S'], help="注意力层的类型序列")
    model_group.add_argument('--no_mlp', action='store_true', help='如果设置，则不在注意力块后使用MLP')
    model_group.add_argument('--dropout', type=float, default=0.1, help='模型中使用的dropout率')

    # --- 训练过程参数 ---
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--lr', type=float, default=5e-4, help='学习率 (推荐使用较小值)')
    train_group.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    train_group.add_argument('--epochs', type=int, default=500, help='训练轮数')
    train_group.add_argument('--eval_interval', type=int, default=10, help='评估间隔')
    train_group.add_argument('--neg_sample_k', type=int, default=128, help='每个正样本对使用的负样本数量')
    train_group.add_argument('--temperature', type=float, default=0.07, help='InfoNCE loss 中的温度系数 (推荐使用较小值)')
    train_group.add_argument('--grad_clip_norm', type=float, default=1.0, help='梯度裁剪的范数阈值')

    # --- 运行设备参数 ---
    run_group = parser.add_argument_group('Runtime Parameters')
    run_group.add_argument('--device', type=str, default='auto', help="device ('auto', 'cpu', 'cuda')")
    args = parser.parse_args()
    if len(args.hidden_dims) != len(args.num_heads) or len(args.hidden_dims) != len(args.layer_types):
        raise ValueError("`--hidden_dims`, `--num_heads`, 和 `--layer_types` 的列表长度必须完全一致！")
    if args.device == 'auto': args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def get_negative_samples(pos_pairs, num_nodes_net2, k, device):
    pos_src = pos_pairs[:, 0].unsqueeze(1)
    neg_candidates = torch.randint(0, num_nodes_net2, (pos_pairs.shape[0], k), device=device)
    return pos_src.expand(-1, k).flatten(), neg_candidates.flatten()

def info_nce_loss(pos_sim, neg_sim, temperature):
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

@torch.no_grad()
def evaluate(model, data, split='val'):
    model.eval()
    all_embeddings = model(data['features'], data['edge_index'], data['batch_mapping'], data)
    embeds1, embeds2 = all_embeddings[:data['num_nodes_net1']], all_embeddings[data['num_nodes_net1']:]
    pairs_to_eval = data[f'{split}_pairs']
    source_embeds = embeds1[pairs_to_eval[:, 0]]
    sim_matrix = cosine_similarity(source_embeds.cpu().numpy(), embeds2.cpu().numpy())
    ranks = []
    for i in range(len(pairs_to_eval)):
        true_target_id = pairs_to_eval[i, 1].item()
        sorted_indices = np.argsort(-sim_matrix[i, :])
        rank = np.where(sorted_indices == true_target_id)[0][0]
        ranks.append(rank + 1)
    ranks = np.array(ranks)
    return np.mean(1.0 / ranks), np.mean(ranks <= 1), np.mean(ranks <= 10)

def main(args):
    data = load_and_prepare_data(args)
    model = NodeEmbeddingModel(
        semantic_dim=data['semantic_dim'], topo_dim=data['topo_dim'], struct_dim=data['struct_dim'],
        embed_dim=args.embed_dim, hidden_dims=args.hidden_dims, num_heads=args.num_heads,
        layer_types=args.layer_types, use_mlp=not args.no_mlp, dropout=args.dropout
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_mrr, best_model_state = 0.0, None
    print("\n--- 开始训练 ---")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        base_embeds, projected_embeds = model(data['features'], data['edge_index'], data['batch_mapping'], data)
        
        embeds1, embeds2 = projected_embeds[:data['num_nodes_net1']], projected_embeds[data['num_nodes_net1']:]
        pos_pairs = data['train_pairs']
        
        pos_src_embeds, pos_tgt_embeds = embeds1[pos_pairs[:, 0]], embeds2[pos_pairs[:, 1]]
        pos_sim = torch.sum(F.normalize(pos_src_embeds) * F.normalize(pos_tgt_embeds), dim=1)

        neg_src_idx, neg_tgt_idx = get_negative_samples(pos_pairs, data['num_nodes_net2'], args.neg_sample_k, args.device)
        neg_src_embeds, neg_tgt_embeds = embeds1[neg_src_idx], embeds2[neg_tgt_idx]
        neg_sim_flat = torch.sum(F.normalize(neg_src_embeds) * F.normalize(neg_tgt_embeds), dim=1)
        neg_sim = neg_sim_flat.view(pos_pairs.shape[0], args.neg_sample_k)
        
        loss = info_nce_loss(pos_sim, neg_sim, args.temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            mrr, h1, h10 = evaluate(model, data, split='val')
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val MRR: {mrr:.4f} | Val H@1: {h1:.4f} | Val H@10: {h10:.4f}")
            
            # --- 手动打印学习率变化 ---
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(mrr)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"学习率已降低至: {new_lr:.6f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                best_model_state = model.state_dict()
                print(f"🚀 New best model found at epoch {epoch+1} with MRR: {best_mrr:.4f}")

    if best_model_state:
        print("\n--- 加载最佳模型并在测试集上进行最终评估 ---")
        model.load_state_dict(best_model_state)
        test_mrr, test_h1, test_h10 = evaluate(model, data, split='test')
        print("\n--- The final result ---")
        print(f"  - Test MRR:     {test_mrr:.4f}")
        print(f"  - Test Hits@1:  {test_h1:.4f}")
        print(f"  - Test Hits@10: {test_h10:.4f}")

if __name__ == '__main__':
    args = get_args()
    main(args)
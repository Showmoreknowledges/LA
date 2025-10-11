import torch
from torch import nn
import torch.nn.functional as F # 导入 F
from torch_geometric.utils import to_dense_batch
from esa_core import ESA
from utils import SmallMLP, LN

class NodeEmbeddingModel(nn.Module):
    """
    UPGRADED: 在融合前对多通道特征进行归一化。
    """
    def __init__(self, semantic_dim, topo_dim, struct_dim,
                 embed_dim, hidden_dims, num_heads, layer_types, use_mlp, dropout):
        super(NodeEmbeddingModel, self).__init__()
        
        self.semantic_encoder = nn.Linear(semantic_dim, hidden_dims[0])
        
        self.use_topo = topo_dim > 0
        if self.use_topo:
            self.topo_encoder = nn.Linear(topo_dim, hidden_dims[0])
            
        self.use_struct = struct_dim > 0
        if self.use_struct:
            self.struct_encoder = nn.Linear(struct_dim, hidden_dims[0])
            
        self.norm = LN(hidden_dims[0])

        self.esa = ESA(
            dim_in=hidden_dims[0], hidden_dims=hidden_dims, num_heads=num_heads,
            layer_types=layer_types, use_mlp=use_mlp, mlp_class=SmallMLP,
            norm_class=LN, dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_dims[-1], embed_dim)
        self.projection_head = SmallMLP(in_dim=embed_dim, out_dim=embed_dim, num_layers=2)

    def forward(self, x, edge_index, batch_mapping, data_dict):
        sem_dim = data_dict['semantic_dim']
        topo_dim = data_dict['topo_dim']
        
        # --- 特征切片与独立编码 ---
        semantic_x = x[:, :sem_dim]
        # 【重要修复】对每个编码后的特征进行L2归一化
        h_semantic = F.normalize(self.semantic_encoder(semantic_x), p=2, dim=1)
        fused_h = h_semantic
        
        if self.use_topo:
            topo_x = x[:, sem_dim : sem_dim + topo_dim]
            h_topo = F.normalize(self.topo_encoder(topo_x), p=2, dim=1)
            fused_h = fused_h + h_topo # 现在相加是公平的
            
        if self.use_struct:
            struct_x = x[:, sem_dim + topo_dim:]
            h_struct = F.normalize(self.struct_encoder(struct_x), p=2, dim=1)
            fused_h = fused_h + h_struct

        # --- 后续流程保持不变 ---
        h = self.norm(fused_h)
        h_dense, node_mask = to_dense_batch(h, batch_mapping)
        max_nodes = h_dense.size(1)

        h_dense = self.esa(h_dense, edge_index, batch_mapping, max_nodes)
        
        base_embeds_dense = self.output_proj(h_dense)
        base_embeds = base_embeds_dense[node_mask]
        projected_embeds = self.projection_head(base_embeds)

        if self.training:
            return base_embeds, projected_embeds
        else:
            return base_embeds
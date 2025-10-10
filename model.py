import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from esa_core import ESA
from utils import SmallMLP, LN

class NodeEmbeddingModel(nn.Module):
    """
    封装了ESA核心模块，用于生成节点嵌入的完整模型。
    """
    def __init__(self, in_features, embed_dim, hidden_dims, num_heads, layer_types, use_mlp, dropout):
        super(NodeEmbeddingModel, self).__init__()

        self.node_mlp = SmallMLP(in_features, hidden_dims[0], num_layers=2)
        self.norm = LN(hidden_dims[0])

        self.esa = ESA(
            dim_in=hidden_dims[0],
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            layer_types=layer_types,
            use_mlp=use_mlp,
            mlp_class=SmallMLP,
            norm_class=LN,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_dims[-1], embed_dim)

    def forward(self, x, edge_index, batch_mapping):
        h = self.norm(self.node_mlp(x))
        h_dense, node_mask = to_dense_batch(h, batch_mapping)
        max_nodes = h_dense.size(1)

        h_dense = self.esa(h_dense, edge_index, batch_mapping, max_nodes)
        h_dense = self.output_proj(h_dense)

        return h_dense[node_mask]
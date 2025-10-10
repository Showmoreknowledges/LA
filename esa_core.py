import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.utils import unbatch_edge_index

class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, adj_mask=None):
        batch_size = Q.size(0)
        head_dim = self.dim_V // self.num_heads

        q_proj = self.fc_q(Q)
        k_proj = self.fc_k(K)
        v_proj = self.fc_v(K)

        q_reshaped = q_proj.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k_reshaped = k_proj.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v_reshaped = v_proj.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if adj_mask is not None:
            adj_mask_expanded = adj_mask.unsqueeze(1)
            adj_mask = adj_mask_expanded

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped,
                attn_mask=adj_mask, 
                dropout_p=self.dropout_p if self.training else 0
            )
        
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_V)
        out = out + F.mish(self.fc_o(out))
        return out

class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, dropout):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout)

    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)

def get_adj_mask_from_edge_index_node(edge_index, batch_mapping, max_nodes, device):
    batch_size = int(batch_mapping.max().item() + 1)
    
    adj_mask = torch.ones(
        size=(batch_size, max_nodes, max_nodes),
        device=device,
        dtype=torch.bool,
    )
    
    edge_indices_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    
    for i, graph_edge_index in enumerate(edge_indices_unbatched):
        num_nodes_in_graph = int(torch.max(graph_edge_index)) + 1
        adj_mask[i, :num_nodes_in_graph, :num_nodes_in_graph] = False
        adj_mask[i, graph_edge_index[0, :], graph_edge_index[1, :]] = True
        adj_mask[i, graph_edge_index[1, :], graph_edge_index[0, :]] = True
        adj_mask[i].fill_diagonal_(True) # Mask self-attention

    return adj_mask


class SABComplete(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout, use_mlp, mlp_class, norm_class):
        super(SABComplete, self).__init__()
        self.sab = SAB(dim_in, dim_out, num_heads, dropout)
        self.norm1 = norm_class(dim_in)
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = mlp_class(dim_out, dim_out)
            self.norm2 = norm_class(dim_out)
        
        self.proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else None

    def forward(self, x, adj_mask):
        x_norm = self.norm1(x)
        attn_out = self.sab(x_norm, adj_mask)
        
        x_res = x if self.proj is None else self.proj(x)
        out = x_res + attn_out

        if self.use_mlp:
            out_norm = self.norm2(out)
            mlp_out = self.mlp(out_norm)
            out = out + mlp_out
            
        return out

class ESA(nn.Module):
    def __init__(self, dim_in, hidden_dims, num_heads, layer_types, use_mlp, mlp_class, norm_class, dropout=0.1):
        super(ESA, self).__init__()
        self.layer_types = layer_types
        self.layers = nn.ModuleList()
        
        current_dim = dim_in
        for i, layer_type in enumerate(layer_types):
            self.layers.append(
                SABComplete(
                    dim_in=current_dim,
                    dim_out=hidden_dims[i],
                    num_heads=num_heads[i],
                    dropout=dropout,
                    use_mlp=use_mlp,
                    mlp_class=mlp_class,
                    norm_class=norm_class
                )
            )
            current_dim = hidden_dims[i]

    def forward(self, x, edge_index, batch_mapping, max_nodes):
        adj_mask_m = get_adj_mask_from_edge_index_node(edge_index, batch_mapping, max_nodes, x.device) if 'M' in self.layer_types else None

        for i, layer in enumerate(self.layers):
            current_mask = adj_mask_m if self.layer_types[i] == 'M' else None
            x = layer(x, current_mask)
            
        return x
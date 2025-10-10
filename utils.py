from torch import nn

class LN(nn.Module):
    """Layer Normalization Wrapper"""
    def __init__(self, dim, **kwargs):
        super(LN, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class SmallMLP(nn.Module):
    """A simple two-layer MLP with a ReLU activation."""
    def __init__(self, in_dim, out_dim, inter_dim_ratio=2, num_layers=2):
        super(SmallMLP, self).__init__()
        inter_dim = in_dim * inter_dim_ratio
        layers = []
        layers.append(nn.Linear(in_dim, inter_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(inter_dim, inter_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(inter_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
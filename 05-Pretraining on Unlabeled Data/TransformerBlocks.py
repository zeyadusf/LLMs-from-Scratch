import torch
from torch import nn

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    This implementation follows the approximation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForwardGELU(nn.Module):
    """
    A feed-forward neural network layer with GELU activation.
    
    Args:
        cfg (dict): Configuration dictionary with key 'emb_dim' representing the embedding dimension.
    
    The network consists of:
    - A linear layer projecting the input from 'emb_dim' to 4 * 'emb_dim'
    - GELU activation function
    - A linear layer projecting back to 'emb_dim'
    """
    def __init__(self, cfg):
        super(FeedForwardGELU, self).__init__()
        emb_dim = cfg['emb_dim']
        
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift

import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Args:
        emb_dim (int): The dimension of the input embeddings.
    
    Attributes:
        eps (float): A small value to avoid division by zero.
        scale (nn.Parameter): Learnable scale parameter.
        shift (nn.Parameter): Learnable shift parameter.
    """
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


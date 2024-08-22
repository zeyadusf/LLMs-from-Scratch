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

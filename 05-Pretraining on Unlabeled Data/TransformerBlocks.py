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

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        context_length (int): The length of the input sequence.
        dropout (float): Dropout probability.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): Whether to include bias in query, key, and value projections. Default is False.
    
    Attributes:
        d_out (int): Output dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        w_queries (nn.Linear): Linear projection for queries.
        w_keys (nn.Linear): Linear projection for keys.
        w_values (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.
        dropout (nn.Dropout): Dropout layer.
        mask (torch.Tensor): Lower triangular mask to ensure causality.
    """
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batches, num_tokens, dim_in = x.shape

        # Linear projections
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        # Reshape and transpose for multi-head attention
        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score calculation
        attn_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_score.masked_fill_(mask_bool, float('-inf'))

        # Softmax to get attention weights
        attn_weight = torch.softmax(attn_score / (self.head_dim ** 0.5), dim=-1)
        attn_weight = self.dropout(attn_weight)

        # Context vector computation
        context_vec = (attn_weight @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec

# TODO: Tranformer Model in notebook, then build GPT model
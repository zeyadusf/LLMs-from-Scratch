from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_queries = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.w_values = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(in_features=d_in, out_features=d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )


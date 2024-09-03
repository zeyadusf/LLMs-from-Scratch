# from torch import nn
# import torch
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_in: int, d_out: int, context_length: int,
#                  dropout: float, num_heads: int, qkv_bias: bool = False):
#         super().__init__()
#         assert d_out % num_heads == 0, "d_out must be divisible num_heads"
#         self.d_out = d_out
#         self.num_heads = num_heads
#         self.head_dim = d_out // num_heads
#
#         self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.out_proj = nn.Linear(d_out, d_out)
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer(
#             'mask',
#             torch.tril(torch.ones(context_length, context_length), diagonal=1)
#         )
#
#     def forward(self, x):
#         batches, num_tokens, dim_in = x.shape
#         # Shape: (b, num_tokens, d_out)
#         queries = self.w_queries(x)
#         keys = self.w_keys(x)
#         values = self.w_values(x)
#
#         # We implicitly split the matrix by adding a `num_heads` dimension
#         # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
#         # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
#         keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#         queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#         values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#
#         attn_score = queries @ keys.transpose(2, 3)
#         mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
#         attn_score.masked_fill(mask_bool, -torch.inf)
#
#         attn_weight = torch.softmax(
#             attn_score / keys.shape[-1]**0.5,
#             dim=-1)
#         attn_weight = self.dropout(attn_weight)
#
#         # Shape: (b, num_tokens, num_heads, head_dim)
#         context_vec = (attn_weight @ values).transpose(1, 2)
#         # Combine heads, where self.d_out = self.num_heads * self.head_dim
#         context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)
#         context_vec = self.out_proj(context_vec)  # optional projection
#         return context_vec
#
#
# if __name__ == "__main__":
#     inputs = torch.tensor(
#         [[0.43, 0.15, 0.89],  # Your (x^1)
#          [0.55, 0.87, 0.66],  # journey (x^2)
#          [0.57, 0.85, 0.64],  # starts (x^3)
#          [0.22, 0.58, 0.33],  # with (x^4)
#          [0.77, 0.25, 0.10],  # one (x^ 5)
#          [0.05, 0.80, 0.55]]  # step (x^6)
#     )
#
#     torch.manual_seed(123)
#     batch = torch.stack((inputs, inputs, inputs), dim=0)
#     batch_size, context_length, d_in = batch.shape
#     d_out = 2
#
#     mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
#
#     context_vecs = mha(batch)
#
#     print(context_vecs)
#     print("context_vecs.shape:", context_vecs.shape)


# fixed issue

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
            torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
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
        attn_scores = (queries @ keys.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Apply mask: Broadcasting across batches and heads
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector computation
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec

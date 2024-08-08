import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batches, num_tokens, d_in = x.shape
        keys = self.w_keys(x)
        values = self.w_values(x)
        queries = self.w_queries(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1], dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    d_in = batch.shape[-1]
    d_out = 2
    print(f'batch.shape:{batch.shape} ')
    context_len = batch.shape[1]
    ca_att = CausalAttention(d_in, d_out, context_len, 0.0)
    context_vector = ca_att(batch)
    print(context_vector)
    print(f'context_vec.shape {context_vector.shape}')


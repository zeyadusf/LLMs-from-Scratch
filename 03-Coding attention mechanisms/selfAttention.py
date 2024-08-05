import torch


class SelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.w_queries = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_keys = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_values = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_keys
        queries = x @ self.w_queries
        values = x @ self.w_values

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        return attention_weights @ values  # Context vector.


class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.w_queries = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self,x):
        keys = self.w_keys(x)
        queries = self.w_queries(x)
        values = self.w_values(x)

        attention_score = queries @ keys.T
        attention_weights = torch.softmax(
            attention_score / keys.shape[-1]**0.5,
            dim=-1
        )
        return attention_weights @ values  # Context Vector.


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

    sa1 = SelfAttentionV1(3, 2)
    sa2 = SelfAttentionV2(3, 2)

    print(f'Self attention with Parameter implementation [V1]: \n{sa1(inputs)}')
    print('-*-*'*10)
    print(f'Self attention with Linear implementation [V2]: \n{sa2(inputs)}')




from MultiHeadAttention import MultiHeadAttention
from FeedForwardGELU import FeedForwardGELU
from LayerNorm import LayerNorm
from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        resid_conn = x
        x = self.norm1(x)  # pre-LayerNorm
        x = self.attn(x)
        x = self.dropout(x)
        x = x + resid_conn

        resid_conn = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + resid_conn
        return x


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    trf_block = TransformerBlock(cfg=GPT_CONFIG_124M)
    output = trf_block(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)

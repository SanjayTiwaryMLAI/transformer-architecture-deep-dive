"""
Scaled Dot-Product and Multi-Head Attention from scratch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, heads, seq_len, d_k)
        key:   (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_v)
        mask:  Optional boolean mask — True = keep, False = mask out
    Returns:
        output:  (batch, heads, seq_len, d_v)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    if dropout > 0.0:
        weights = F.dropout(weights, p=dropout)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_out, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout if self.training else 0.0
        )
        B, H, L, D = attn_out.size()
        concat = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(concat)

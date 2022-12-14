import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.utility.scaledproduct import ScaledDotProductAttention

# From https://github.com/jadore801120/attention-is-all-you-need-pytorch


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_k, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q = x.size(0), x.size(1)

        residual = x
        x = self.layer_norm(x)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_q, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_q, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        return q, attn

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.exchangeable_layer import Exchangeable
from layers.mh_attention import MultiHeadAttention

from math import log as mlog


class MHAttentionBody(nn.Module):
    def __init__(self, n_head, hid, hid_att):
        super().__init__()
        self.attention_item = MultiHeadAttention(n_head, hid, hid_att)
        self.attention_agent = MultiHeadAttention(n_head, hid, hid_att)
        self.fc = nn.Linear(2 * hid, hid)

    def forward(self, x):
        residual = x
        bs, na, ni, hid = x.shape

        x_item = x.reshape(-1, ni, hid)
        x_item, _ = self.attention_item(x_item)
        x_item = x_item.reshape(x.shape)

        x_agent = x.permute(0, 2, 1, 3).reshape(-1, na, hid)
        x_agent, _ = self.attention_agent(x_agent)
        x_agent = x_agent.reshape(bs, ni, na, hid).permute(0, 2, 1, 3)

        x = F.tanh(torch.cat([x_item, x_agent], dim=-1))
        x = self.fc(x) + residual
        return x


class AttentionHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.temperature = hid ** 0.5
        self.encoder_item = nn.Linear(hid, hid, bias=False)
        self.encoder_agent = nn.Linear(hid, hid, bias=False)
        self.fc_payment = nn.Sequential(nn.Linear(hid, 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hid, eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)  # [-1, num_agents, num_items, hid]

        x_agent = self.encoder_agent(x).mean(2)  # [-1, num_agents, hid]
        x_item = self.encoder_item(x).mean(1)  # [-1, num_items, hid]

        logits = torch.matmul(x_agent / self.temperature, x_item.transpose(1, 2))  # [-1, num_agents, num_items]

        logits = torch.cat([logits, -logits.sum(1, keepdim=True)], dim=1)  # [-1, num_agents + 1, num_items]

        alloc = F.softmax(logits, 1)[:, :-1]  # [-1, num_agents, num_items]

        pay = self.fc_payment(x_agent).squeeze(-1)  # [-1, num_agents]

        return alloc, pay


class MLPHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.fc_alloc = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.encoder_payment = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.fc_payment = nn.Sequential(nn.Linear(hid, 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hid, eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)  # [-1, num_agents, num_items, hid]

        logits = self.fc_alloc(x).squeeze(-1)  # [-1, num_agents, num_items]

        logits = torch.cat([logits, -logits.sum(1, keepdim=True)], dim=1)  # [-1, num_agents + 1, num_items]

        alloc = F.softmax(logits, 1)[:, :-1]  # [-1, num_agents, num_items]

        x_agent = self.encoder_payment(x).mean(2)  # [-1, num_agents, hid]
        pay = self.fc_payment(x_agent).squeeze(-1)  # [-1, num_agents]

        return alloc, pay


class PartAttentionHead(nn.Module):
    def __init__(self, hid, n_misreports):
        super().__init__()
        self.temperature = hid ** 0.5
        self.n_misreports, self.hid = n_misreports, hid
        self.encoder_item = nn.Linear(hid, hid)
        self.encoder_agent = nn.Linear(hid, hid * n_misreports)

    def forward(self, x):
        x_agent = self.encoder_agent(x).mean(2)  # [-1, num_agents, n_misreports * hid]
        x_agent = x_agent.reshape(
            *x_agent.shape[:2], self.n_misreports, self.hid
        )  # [-1, num_agents, n_misreports, hid]
        x_agent = x_agent.transpose(1, 2).transpose(0, 1)  # [n_misreports, -1 num_agents, hid]

        x_item = self.encoder_item(x).mean(1)  # [-1, num_items, hid]
        x_item = x_item.transpose(1, 2).unsqueeze(0)  # [1, -1, hid, num_items]

        x = torch.matmul(x_agent / self.temperature, x_item)  # [n_misreports, -1, num_agents, num_items]

        return x


class PartMLPHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        return self.fc(x)


class ExchangeableHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.alloc_layer = Exchangeable(hid, 1)
        self.pay_layer = Exchangeable(hid, 1)

    def forward(self, x):
        logits = self.alloc_layer(x).squeeze(-1)  # [-1, num_agents, num_items]
        logits = torch.cat([logits, -logits.sum(1, keepdim=True)], dim=1)  # [-1, num_agents + 1, num_items]

        alloc = F.softmax(logits, 1)[:, :-1]  # [-1, num_agents, num_items]

        pay = torch.sigmoid(self.pay_layer(x).squeeze(-1).mean(2))  # [-1, num_agents]

        return alloc, pay


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, item_wise=True):
        """
        item_wise: True for encoding of items, False for encoding of agents
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-mlog(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, 1, d_model)
        pe[:, 0, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 0, 1::2] = torch.cos(position * div_term)

        self.item_wise = item_wise

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, self.num_agents, self.num_items, self.hid]
        """
        permutation = (2, 1, 0, 3) if self.item_wise else (1, 0, 2, 3)
        x = torch.permute(x, permutation)
        x = x + self.pe[: x.size(0)]
        x = torch.permute(x, permutation)

        return x

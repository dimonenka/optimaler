from torch import nn
from torch.nn import functional as F

from layers.blocks import AttentionHead, MHAttentionBody, MLPHead, PositionalEncoding
from layers.exchangeable_layer import Exchangeable
from core.nets.additive_net import AdditiveNet


class AdditiveNetAttention(AdditiveNet):
    def __init__(self, model_config, device):
        super().__init__(model_config, device)

    def init(self):
        self.hid = self.config.net.hid
        self.hid_att = self.config.net.hid_att
        self.n_layers = self.config.net.n_attention_layers
        self.n_heads = self.config.net.n_attention_heads

        if self.config.net.activation_att.lower() == 'relu':
            self.activation = F.relu
        elif self.config.net.activation_att.lower() == 'tanh':
            self.activation = F.tanh
        else:
            raise NotImplementedError

        self.create_layers()

    def create_layers(self):
        self.create_input_layers()
        self.create_body_layers()
        self.create_head_layers()

    def create_input_layers(self):
        self.input_layer = Exchangeable(1, self.hid, add_channel_dim=True).to(self.device)

        if self.config.net.pos_enc:
            if self.config.net.pos_enc_part > 1:
                pos_enc_part_layer = PositionalEncoding(self.config.net.pos_enc_part, self.hid, item_wise=False)
                self.input_layer = nn.Sequential(self.input_layer, pos_enc_part_layer)
            if self.config.net.pos_enc_item > 1:
                pos_enc_item_layer = PositionalEncoding(self.config.net.pos_enc_item, self.hid, item_wise=True)
                self.input_layer = nn.Sequential(self.input_layer, pos_enc_item_layer)

    def create_body_layers(self):
        self.body_layers = nn.ModuleList(
            [MHAttentionBody(self.n_heads, self.hid, self.hid_att) for _ in range(self.n_layers)]
        ).to(self.device)

    def create_head_layers(self):
        self.head_layer = AttentionHead(hid=self.hid).to(self.device)

    def forward(self, x, return_intermediates=False):
        valuations = x  # [-1, self.num_agents, self.num_items]

        x = self.activation(self.input_layer(x))  # [-1, self.num_agents, self.num_items, self.hid]

        for layer in self.body_layers:
            x = self.activation(layer(x))  # [-1, self.num_agents, self.num_items, self.hid]

        alloc, pay = self.head_layer(x)

        matrix_dot = (alloc * valuations).sum(dim=-1)
        final_pay = pay * matrix_dot

        if return_intermediates:
            return alloc, final_pay, pay
        return alloc, final_pay

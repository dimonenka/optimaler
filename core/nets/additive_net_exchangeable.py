from torch import nn
from torch.nn import functional as F

from layers.blocks import ExchangeableHead
from layers.exchangeable_layer import Exchangeable
from core.nets.additive_net_attention import AdditiveNetAttention


class AdditiveNetExchangeable(AdditiveNetAttention):
    def __init__(self, model_config, device):
        super().__init__(model_config, device)

    def init(self):
        self.hid = self.config.net.hid_exch
        self.n_layers = self.config.net.n_exch_layers

        if self.config.net.activation_exch.lower() == 'relu':
            self.activation = F.relu
        elif self.config.net.activation_exch.lower() == 'tanh':
            self.activation = F.tanh
        else:
            raise NotImplementedError

        self.create_layers()

    def create_input_layers(self):
        self.input_layer = Exchangeable(1, self.hid, add_channel_dim=True).to(self.device)

    def create_body_layers(self):
        self.body_layers = nn.ModuleList([Exchangeable(self.hid, self.hid) for _ in range(self.n_layers - 2)]).to(
            self.device
        )

    def create_head_layers(self):
        self.head_layer = ExchangeableHead(self.hid).to(self.device)

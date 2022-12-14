import torch
from torch import nn
from torch.nn import functional as F


class AdditiveNet(nn.Module):
    def __init__(self, model_config, device):
        super(AdditiveNet, self).__init__()
        self.config = model_config
        self.device = device
        self.init()

    def init(self):
        self.alloc_layers = nn.ModuleList([])
        self.pay_layers = nn.ModuleList([])

        self.create_constants()
        self.create_allocation_layers()
        self.create_payment_layers()

    def create_constants(self):
        if self.config.net.init is "None":
            init = None
        elif self.config.net.init == "gu":
            init = nn.init.xavier_uniform_
        elif self.config.net.init == "gn":
            init = nn.init.xavier_normal_
        self.init_weights = init

        if self.config.net.activation == "tanh":
            activation = nn.Tanh()
        elif self.config.net.activation == "relu":
            activation = nn.ReLU()
        self.activation = activation

        self.num_agents = self.config.num_agents
        self.num_items = self.config.num_items

        self.num_a_layers = self.config.net.num_a_layers
        self.num_p_layers = self.config.net.num_p_layers

        self.num_a_hidden_units = self.config.net.num_a_hidden_units
        self.num_p_hidden_units = self.config.net.num_p_hidden_units

        self.num_in = self.num_agents * self.num_items
        self.num_a_output = (self.num_agents + 1) * self.num_items

        self.ln = self.config.net.layer_norm

    def create_allocation_layers(self):
        self.create_input_alloc_layer()
        self.create_body_alloc_layer()
        self.create_head_alloc_layer()
        if self.ln:
            self.create_ln_alloc_layers()

    def create_input_alloc_layer(self):
        alloc_first_layer = nn.Linear(self.num_in, self.num_a_hidden_units).to(self.device)
        self.init_weights(alloc_first_layer.weight)
        nn.init.zeros_(alloc_first_layer.bias)
        self.alloc_layers.append(alloc_first_layer)

    def create_body_alloc_layer(self):
        for i in range(1, self.num_a_layers - 1):
            alloc_new_layer = nn.Linear(self.num_a_hidden_units, self.num_a_hidden_units).to(self.device)
            self.init_weights(alloc_new_layer.weight)
            nn.init.zeros_(alloc_new_layer.bias)
            self.alloc_layers.append(alloc_new_layer)

    def create_head_alloc_layer(self):
        alloc_output_layer = nn.Linear(self.num_a_hidden_units, self.num_a_output).to(self.device)
        self.init_weights(alloc_output_layer.weight)
        nn.init.zeros_(alloc_output_layer.bias)
        self.alloc_layers.append(alloc_output_layer)

    def create_ln_alloc_layers(self):
        self.a_lns = nn.ModuleList([])
        for i in range(self.num_a_layers - 1):
            layer = nn.LayerNorm(self.num_a_hidden_units, eps=1e-3).to(self.device)
            self.a_lns.append(layer)

    def create_payment_layers(self):
        self.create_input_payment_layer()
        self.create_body_payment_layer()
        self.create_head_payment_layer()
        if self.ln:
            self.create_ln_payment_layers()

    def create_input_payment_layer(self):
        pay_first_layer = nn.Linear(self.num_in, self.num_p_hidden_units).to(self.device)
        self.init_weights(pay_first_layer.weight)
        nn.init.zeros_(pay_first_layer.bias)
        self.pay_layers.append(pay_first_layer)

    def create_body_payment_layer(self):
        for i in range(1, self.num_p_layers - 1):
            pay_new_layer = nn.Linear(self.num_p_hidden_units, self.num_p_hidden_units).to(self.device)
            self.init_weights(pay_new_layer.weight)
            nn.init.zeros_(pay_new_layer.bias)
            self.pay_layers.append(pay_new_layer)

    def create_head_payment_layer(self):
        pay_output_layer = nn.Linear(self.num_p_hidden_units, self.num_agents).to(self.device)
        self.init_weights(pay_output_layer.weight)
        nn.init.zeros_(pay_output_layer.bias)
        self.pay_layers.append(pay_output_layer)

    def create_ln_payment_layers(self):
        self.p_lns = nn.ModuleList([])
        for i in range(self.num_p_layers - 1):
            layer = nn.LayerNorm(self.num_p_hidden_units, eps=1e-3).to(self.device)
            self.p_lns.append(layer)

    def forward(self, x, return_intermediates=False):
        x_in = x.view([-1, self.num_in])  # reshape to vector

        alloc = self.forward_th_allocation(x_in)
        pay = self.forward_th_payment(x_in)

        # final layer
        matrix_dot = (alloc * x).sum(dim=-1)
        final_pay = pay * matrix_dot

        if return_intermediates:
            return alloc, final_pay, pay
        return alloc, final_pay

    def forward_th_allocation(self, x):
        alloc = self.alloc_layers[0](x)
        if self.ln:
            alloc = self.a_lns[0](alloc)
        alloc = self.activation(alloc)
        for i in range(1, self.num_a_layers - 1):
            alloc = self.alloc_layers[i](alloc)
            if self.ln:
                alloc = self.a_lns[i](alloc)
            alloc = self.activation(alloc)

        alloc = self.alloc_layers[-1](alloc)
        alloc = F.softmax(alloc.view([-1, self.num_agents + 1, self.num_items]), dim=1)
        alloc = alloc[:, :-1, :]

        return alloc

    def forward_th_payment(self, x):
        pay = self.pay_layers[0](x)
        if self.ln:
            pay = self.p_lns[0](pay)
        pay = self.activation(pay)
        for i in range(1, self.num_p_layers - 1):
            pay = self.pay_layers[i](pay)
            if self.ln:
                pay = self.p_lns[i](pay)
            pay = self.activation(pay)

        pay = self.pay_layers[-1](pay)
        pay = torch.sigmoid(pay)

        return pay

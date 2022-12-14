import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Auction params
__C.num_agents = 2
__C.num_items = 5

# RegretNet
__C.net.num_a_layers = 3
__C.net.num_p_layers = 3

# EquivariantNet
__C.net.n_exch_layers = 6

# RegretFormer
__C.net.n_attention_layers = 2
__C.net.n_attention_heads = 4
__C.net.hid_equiv = 128
__C.net.hid_att = 32
__C.net.hid = 128

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_01"
__C.min = 0
__C.max = 1

import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Auction params
__C.num_agents = 2
__C.num_items = 2

# EquivariantNet
__C.net.n_exch_layers = 3

# RegretFormer
__C.net.n_attention_heads = 2
__C.net.hid_att = 32
__C.net.hid = 64

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_01"
__C.min = 0
__C.max = 1

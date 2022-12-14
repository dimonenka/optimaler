import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Plot
__C.plot.bool = True

# Auction params
__C.num_agents = 1
__C.num_items = 2

# RegretFormer
__C.net.pos_enc = True
__C.net.pos_enc_part = 1
__C.net.pos_enc_item = 2

# Distribution type - 'uniform_01' or 'uniform_416_47'
__C.distribution_type = "uniform_416_47"
__C.min = 4
__C.max = 16

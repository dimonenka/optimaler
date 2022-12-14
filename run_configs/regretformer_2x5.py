from copy import deepcopy

from core.configs.additive_2x5_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "additive_2x5_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet
__C.architecture = "RegretFormer"

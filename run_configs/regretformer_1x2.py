from copy import deepcopy

from core.configs.additive_1x2_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "additive_1x2_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet
__C.architecture = "RegretFormer"

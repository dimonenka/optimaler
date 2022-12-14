from copy import deepcopy

from core.configs.additive_2x2_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "additive_2x2_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet
__C.architecture = "EquivariantNet"

# Type of target pretrained net
cfg.distill.architecture = "RegretFormer"

# Type of procedure - 'standard' (train net), 'cross_val' or 'distillation'
__C.regret_type = "distillation"

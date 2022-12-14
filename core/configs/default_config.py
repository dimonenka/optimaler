from easydict import EasyDict as edict

__C = edict()
cfg = __C

# 'standard' for DSIC with misreport estimation through gradient descent of inputs
# 'distillation' for distilling a trained network onto a new network (for validation)
# 'cross-val' for cross-validating trained networks (for validation)
__C.regret_type = "standard"

# Type of net - RegretNet, RegretFormer or EquivariantNet
__C.architecture = None

# Bid distribution
__C.distribution_type = None
__C.min = None
__C.max = None

# Auction params
__C.num_agents = None
__C.num_items = None

# Save data for restore.
__C.save_data = "runs"

# Plots
__C.plot = edict()
__C.plot.bool = False
__C.plot.n_points = 201

# Distillation procedure
__C.distill = edict()
__C.distill.architecture = None
__C.distill.validate_target_misreports = True
__C.distill.train_misreports = True

# RegretNet parameters
__C.net = edict()
# initialization g - glorot, h - he + u - uniform, n - normal [gu, gn, hu, hn]
__C.net.init = "gu"
# activations ["tanh", "sigmoid", "relu"]
__C.net.activation = "tanh"
# num_a_layers, num_p_layers - total number of hidden_layers + output_layer, [a - alloc, p - pay]
__C.net.num_a_layers = 3
__C.net.num_p_layers = 3
# num_p_hidden_units, num_p_hidden_units - number of hidden units, [a - alloc, p - pay]
__C.net.num_p_hidden_units = 100
__C.net.num_a_hidden_units = 100
__C.net.layer_norm = False

# RegretFormer parameters
__C.net.hid_att = 16
__C.net.hid = 32
__C.net.n_attention_layers = 1
__C.net.n_attention_heads = 2
__C.net.activation_att = 'tanh'
__C.net.pos_enc = False
__C.net.pos_enc_part = 1
__C.net.pos_enc_item = 1

# EquivariantNet parameters
__C.net.n_exch_layers = 3
__C.net.hid_exch = 32
__C.net.activation_exch = 'relu'

# Train parameters
__C.train = edict()

# Random seed
__C.train.seed = 42
# Iter from which training begins. If restore_iter = 0 for default. restore_iter > 0 for starting
# training form restore_iter [needs saved model]
__C.train.restore_iter = 0
# max iters to train
__C.train.max_iter = 200000
# Learning rate of network param updates
__C.train.learning_rate = 1e-3

""" Train-data params """
# Choose between fixed and online. If online, set adv_reuse to False
__C.train.data = "fixed"
# Number of batches
__C.train.num_batches = 1250
# Train batch size
__C.train.batch_size = 512

""" Train-misreport params """
# Cache-misreports after misreport optimization
__C.train.adv_reuse = True
# Number of misreport initialization for training
__C.train.num_misreports = 1
# Number of steps for misreport computation
__C.train.gd_iter = 50
# Learning rate of misreport computation
__C.train.gd_lr = 0.1
__C.train.gd_lr_step = 1

""" Lagrange Optimization params """
__C.train.w_rgt_init_val = 1
__C.train.rgt_target_start = 0.01
__C.train.rgt_target_end = 0.001
__C.train.rgt_lr = 0.5

""" train summary and save params"""
# Frequency at which models are saved
__C.train.save_iter = 100000
# Train stats print frequency
__C.train.print_iter = 1000

""" Validation params """
__C.val = edict()
# Number of misreport initialization for validation
__C.val.num_misreports = 1
# Number of steps for misreport computation
__C.val.gd_iter = 1000
# Learning rate for misreport computation
__C.val.gd_lr = 0.1
__C.val.gd_lr_step = 1
# Number of validation batches
__C.val.num_batches = 128
__C.val.batch_size = 32
# Frequency at which validation is performed
__C.val.print_iter = 25000
# Validation data frequency
__C.val.data = "online"

# Compute number of samples
__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size

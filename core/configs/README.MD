# Description of configuration parameters

## Configuration file structure

1. The base configuration file is `default_config`. It describes all the parameters for running experiments and is used as a starting point of all higher-level configs. Some of the parameters are overwritten in the higher-level configs.
2. Auction-type configuration files - all other configs in this folder. They overwrite some parameters from the base config by describing a specific auction -- number of items and participants, bid distribution type, and neural network hyperparameters.
3. Ready-to-run configuration files - all configs from `run_configs` folder. They overwrie some parameters from the auction-type configs, i.e. the type of model to use (`architecture`) and what type of experiment to run (`regret_type`).


## Parameters
`regret_type` - type of the experiment. Use `standard` for training an auction network (DSIC with misreport estimation through gradient descent of inputs), 'distillation' for distilling a trained network onto a new network (for validation), or 'cross-val' for cross-misreport validating trained networks (for validation) [Default: `"standard"`, specified in ready-to-run configuration files]

`architecture` - type of the network: "RegretNet", "RegretFormer", or "EquivariantNet". [Default: None, specified in ready-to-run config]

`setting` - the name of the setting. Used for logging (as a folder name) when training the network (`regret_type=standard`) or when loading models for distillation and cross-validation (`regret_type=distillation/cross-val`). If None, the name of the configuration file is used. [Default: None, specified in ready-to-run config]


#### Auction parameters

`num_agents` - the number of participants in an auction [Default: None, specified in auction-type config]

`num_items` - the number of items in an auction [Default: None, specified in auction-type config]

`distribution_type` - type of the bid distribution, "uniform_01" (uniform from 0 to 1 for all items and participants) or "uniform_416_47" (Daskalakis et al. (2012) asymmetric auction) [Default: None, specified in auction-type config]

`min` - min possible value of bid [Default: None, specified in auction-type config]

`max` - max possible value of bid [Default: None, specified in auction-type config]


#### Train process parameters

`train.seed` - random seed fixation [Default: 42]

`train.restore_iter` - iter from which training begins. Set restore_iter = 0 for random network initialization. Set restore_iter > 0 for starting training from restore_iter (requires a saved model) [Default: 0]

`train.max_iter` - number of training iterations (1 iteration = optimizer step on 1 batch) [Default: 200000]

`train.learning_rate` - learning rate for optimizer [Default: 1e-3]

`train.data` - fixed or online generated data. If online, set adv_reuse to False [Default: "fixed"]

`train.num_batches` - number of batches sampled to generate the training dataset [Default: 1250]

`train.batch_size` - size of a single training batch [Default: 512]

`train.adv_reuse` - cache misreports after misreport optimization [Default: True]

`train.num_misreports` - number of misreport initializations during training [Default: 1]

`train.gd_iter` - number of gradient steps for misreport computation during training [Default: 50]

`train.gd_lr` - learning rate of gradient descent for misreport computation during training [Default: 0.1]

`train.gd_lr_step` - learning rate step size (scheduler) for misreport computation during training; no scheduling if set to one [Default: 1]

`train.w_rgt_init_val` - initial value of the regret weight in the loss function (Lagrangian coefficient $\gamma$) [Default: 1]

`train.rgt_target_start` - target regret value (budget) at the beginning of training [Default: 0.01]

`train.rgt_target_end` - target regret value (budget) by the end of training; the budget exponentially converges from rgt_target_start to rgt_target_end during first 2/3 of training; set rgt_target_start to rgt_target_end to turn target regret scheduling off (may hinder performance for low rgt_target_end values) [Default: 0.001]

`train.rgt_lr` - learning rate of dual gradient descent; specifies how quickly Lagrange coefficient $\gamma$ is updated for the network to match the regret budget; values in [0.1, 1] should work fine [Default: 0.5]

`train.save_iter` - frequency at which the models are saved [Default: 100000]

`train.print_iter` - print frequency of statistics during training [Default: 1000]


#### Validation params

`val.num_misreports` - number of misreport initializations during validation [Default: 1]

`val.gd_iter` - number of gradient steps for misreport computation during validation (typically higher than during training for higher precision) [Default: 1000]

`val.gd_lr` - learning rate of gradient descent for misreport computation during validation [Default: 0.1]

`val.gd_lr_step` - learning rate step size (scheduler) for misreport computation during valudation [Default: 1]

`val.num_batches` - number of batches sampled to generate the validation dataset [Default: 128]

`val.batch_size` - size of a single validation batch [Default: 32]

`val.print_iter` - frequency at which validation is performed and the results are printed [Default: 25000]

`val.data` - fixed data or online generated. If online, set adv_reuse to False [Default: "online"]


#### Neural networks architecture
All neural networks parameters have default values specified in `default.cfg` and might be rewritten in auction-type configs.

##### Common parameters

`net.init` - weights initialization, (g - glorot, h - he) + (u - uniform, n - normal) [Default: "gu"]

##### RegretFormer

`net.hid_att` - dimensionality of linear projections of each head in multi-head self-attention [Default: 16]

`net.hid` - hidden size of all layers in RegretFormer [Default: 32]

`net.n_attention_layers` - number of attention layers [Default: 1]

`net.n_attention_heads` - number of heads in each attention layer [Default: 2]

`net.activation_att` - activation function ("relu" or "tanh") [Default: "tanh"]

`net.pos_enc` - boolean, whether to apply positional encoding; should be used to learn auctions with asymmetric participants or items; PE from Vaswani et al. (2017) is applied [Default: False]

`net.pos_enc_part` - Number of participants to encode the positions of; should be set to (the maximal) number of participants to learn auctions with asymmetric participants [Default: 1]

`net.pos_enc_item` - Number of items to encode the positions of; should be set to (the maximal) number of items to learn auctions with asymmetric items [Default: 1]


##### RegretNet

`net.activation` - activation function ("tanh", "sigmoid", "relu") [Default: "tanh"]

`net.num_a_layers` - number of fully-connected layers in the allocation network [Default: 3]

`net.num_a_hidden_units` - hidden size of each allocation layer [Default: 100]

`net.num_p_layers` - number of fully-connected layers in the payment network  [Default: 3]

`net.num_p_hidden_units` - hidden size of each payment layer [Default: 100]

`net.layer_norm` - boolean, whether to apply nn.LayerNorm to outputs of each layer [Default: False]


##### EquivariantNet

Note: changing the activation function or the network size may hinder the performance of EquivariantNet.

`net.n_exch_layers` - the number of exchangeable layers [Default: 3]

`net.hid_exch` - the hidden size of each exchangeable layer [Default: 32]

`net.activation_exch` - activation function ("relu" or "tanh") [Default: "relu"]


#### Distillation parameters

`distill.architecture` - architecture of a pretrained network used as a teacher (target) network [Default: None, specified in ready-to-run confifs]

`distill.validate_target_misreports` - whether to use the optimal misreports approximated for the teacher (target) network during estimating the similarity of outputs of the two networks [Default: True]

`distill.train_misreports` - whether to use the optimal misreports approximated for the student network as an additional training sample (to estimate divergence from the outputs of the teacher network) [Default: True]


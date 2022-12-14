# [Optimal-er Auctions through Attention](https://arxiv.org/abs/2202.13110)
**Dmitry Ivanov, Iskander Safiulin, Igor Filippov, Ksenia Balabaeva**

*RegretNet is a recent breakthrough in the automated design of revenue-maximizing auctions. It combines the flexibility of deep learning with the regret-based approach to relax the Incentive Compatibility (IC) constraint (that participants prefer to bid truthfully) in order to approximate optimal auctions. We propose two independent improvements of RegretNet. The first is a neural architecture denoted as RegretFormer that is based on attention layers. The second is a loss function that requires explicit specification of an acceptable IC violation denoted as regret budget. We investigate both modifications in an extensive experimental study that includes settings with constant and inconstant number of items and participants, as well as novel validation procedures tailored to regret-based approaches. We find that RegretFormer consistently outperforms RegretNet in revenue (i.e. is *optimal-er*) and that our loss function both simplifies hyperparameter tuning and allows to unambiguously control the revenue-regret trade-off by selecting the regret budget.*


This is the official implementation of RegretFormer and the budget-based training procedure in PyTorch. The repository contains the code necessary to reproduce all experiments from the paper, including implementations of the existing architectures RegretNet (Dutting et al., 2019) and EquivariantNet (Rahme et al., 2021). Note that these architectures are also trained with the proposed budget-based procedure, as opposed to the original procedure where the Lagrange coefficients are hand-selected.

## Preparation
Create a python environment and activate it. To install requirements, execute the command:
`pip install -r requirements.txt`

## Run experiments
To run experiments, execute the command:

`python main.py --setting=path/to/setting.py --n_run=number_of_runs --n_gpu=number_of_gpus`

where

`setting` - the path to the setting file (from the repository or customized)

`n_run` - number of runs. In the paper, the experiments were run 3 times [Default: 3]

`n_gpu` - number of gpus [Default: 0]

#### Note on gpus and paralelization

Our code supports paralelization of random seeds through Ray implemented in `main.py`. By default, all experiments are run sequentially on CPUs if `n_gpu` = 0, sequentially on GPUs if 0 < `n_gpu` < `n_run`, and in parallel on GPUs if `n_gpu` == `n_run` (with each random seed being run in a separate process with 1 cpu and 1 gpu allocated). Setting `n_gpu` > `n_run` is the same as setting `n_gpu` == `n_run`. If you want to run experiments on CPUs in parallel, or if you want to allocate several CPUs and/or GPUs to each process, please modify `main.py`.

### Experiments from the paper
Configuration files are provided in the repository for each experiment from the paper.
These are in the `run_configs` folder and are based on configs from `core/configs`.
For futher information about configs, see `core/configs/README.MD`.

Here are some examples for running experiments from the paper (`/` denotes `or`):
1. `regretformer/regretnet/equivariantnet_1x2.py` - RegretFormer/RegretNet/EquivariantNet for the 1x2 (1 bidder, 2 items) setting.
2. `regretformer_2x2/2x3/2x5/3x10.py` - similar to the previous but for other settings.
3. `distillation_regretnet_on_regretformer_1x2.py` - distillation validation procedure for training RegretNet on a pretrained RegretFormer.
4. `cross_val_1x2.py` - cross-misreport validation procedure for all three architectures simultaniously (RegretNet, RegretFormer, Equivariantnet) that have been trained on the 1x2 setting.
5. TODO Multi-settings

### Custom experiments

To run custom experiments, please see the readme file about the configs, and then change the necessary parameters according to your goals.
You can create your custom configuration file in `run_configs` folder - `my_setting.py` - and just as described above, run it using the command:

`python main.py --setting=run_configs/my_setting.py`

## Logging and saving

During execution, the relevant statistics are both printed to stdout and logged as Tensorboard files. These include wall-clock time, loss function, Lagrange coefficient $\gamma$, revenue during training and validation (which should be similar), and regret during training and validation (the latter is typically be higher due to more precise misreport estimation). The parameters of neural networks are also saved regularly. The path where logs and models are saved is specified as cfg.save_data in `core/configs/default_config.py` and `run_train.py`. To plot logs with tensorboard, install it and run:

`tensorboard --logdir runs`

Additionally, the folder `utils` contains the code that was used to transform the Tensorboard logs into tables and plots used in the paper (`extract_csv.py` produces csv tables with raw data and `process_csv.py` aggregates these into final plots and csv tables).

Note that the trained models are provided in the repo (folder `target_nets`) for all three architectures (RegretNet, EquivariantNet, RegretFormer) and all five settings (1x2, 2x2, 2x3, 2x5, 3x10), three seeds each, $R_{max} = 10^{-3}$. These are ready to be used for cross_val and distillation experiments.

## Tutorial

TODO


## Acknowledgments

This code borrows some snippets from [Optimal Auctions through Deep Learning by saisrivatsan
](https://github.com/saisrivatsan/deep-opt-auctions).


## Citation

If you use this repository for your paper, please cite:


`@inproceedings{ivanov2022optimal,`  
  `title={Optimal-er Auctions through Attention},`  
  `author={Ivanov, Dmitry and Safiulin, Iskander and Filippov, Igor and Balabaeva, Ksenia},`  
  `booktitle = {Advances in Neural Information Processing Systems},`  
  `volume = {35},`  
  `year={2022}}`

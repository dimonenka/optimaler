from core import run_train
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str,
                    help="The type of procedure corresponding to the name of config")
parser.add_argument("--n_run", type=int, default=3,
                    help="The number of runs of the same experiment with different random seeds")
parser.add_argument("--n_gpu", type=int, default=0,
                    help="The number of GPUs")
args = parser.parse_args()


RAY = args.n_gpu >= args.n_run
if RAY:
    import ray
    from torch import cuda

    N_GPUS = 1 if cuda.is_available() else 0

    @ray.remote(num_cpus=1, num_gpus=N_GPUS)
    def main_remote(setting, seed):
        return run_train.main(setting, seed)

    ray.init()
    result_ids = []
    for seed in range(args.n_run):
        result_ids.append(main_remote.remote(args.setting, seed))
    ray.get(result_ids)

else:
    for seed in range(args.n_run):
        run_train.main(args.setting, seed)

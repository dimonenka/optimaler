import subprocess
import os

# Clipping valuation values functions
from core.clip_ops.clip_ops import *

# Data generator classes
from core.data import *


CLIPS = {
    'uniform_01': lambda x: clip_op_01(x),
    'uniform_416_47': lambda x: clip_op_416_47(x)
}

GENERATORS = {
    'uniform_01': uniform_01_generator.Generator,
    'uniform_416_47': uniform_416_47_generator.Generator
}


def get_path_and_file(setting_path):
    path, file = os.path.split(setting_path)
    return path, os.path.splitext(file)[0]


def get_objects(setting_path):
    '''
    Get objects from configuration file
    '''
    path, setting_name = get_path_and_file(setting_path)
    import_obj = __import__(path, fromlist=[setting_name])
    cfg = getattr(import_obj, setting_name).cfg
    clip_op = CLIPS[cfg.distribution_type]
    generator = GENERATORS[cfg.distribution_type]
    return cfg, clip_op, generator, setting_name


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8"
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

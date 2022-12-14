import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


"""
Adapted from https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv/52095336
"""


def tabulate_events(dirs):
    # summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for directory in os.listdir(dpath)]
    summary_iterators = [EventAccumulator(directory).Reload() for directory in dirs]

    tags = summary_iterators[0].Tags()['scalars']
    print('tags:', tags)

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)

    steps = [e.step for e in summary_iterators[0].Scalars('Validation/revenue')]
    print('len of steps', len(steps))
    for tag in ['Train/regret', 'Train/revenue', 'Train/w_rgt', 'Validation/regret_grad', 'Validation/revenue']:

        scalars = [acc.Scalars(tag) for acc in summary_iterators]
        scalars_new = []

        for scalar in scalars:
            if len(scalar) > len(steps):
                freq = len(scalar) // len(steps)
                scalar = scalar[freq-1::freq]
            elif len(scalar) < len(steps):
                scalar = [scalar[(i * len(scalar)) // len(steps)] for i in range(len(steps))]
            assert len(steps) == len(scalar), f"len(steps) = {len(steps)}, len(scalar) = {len(scalar)}"
            scalars_new.append(scalar)

        for events in zip(*scalars_new):
            # assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    # dirs = os.listdir(dpath)
    dirs = []
    for root, subdirectories, files in os.walk(dpath):
        for subdirectory in subdirectories:
            if subdirectory.startswith('seed'):
                dirs.append(os.path.join(root, subdirectory))
    print(dirs[0])

    d, steps = tabulate_events(dirs)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    print('total shape', np_values.shape)

    idx_names = ['network', 'setting', 'seed']
    multi_idx = pd.MultiIndex.from_tuples([tuple(directory.split('/')[-3:]) for directory in dirs], names=idx_names)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index].T, index=multi_idx, columns=steps)
        df.sort_index(inplace=True)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path = "runs/ready/"
    to_csv(path)

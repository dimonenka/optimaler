import os

import numpy as np


class BaseGenerator(object):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size

    def build_generator(self, X=None, ADV=None):
        if self.mode is "train":
            if self.config.train.data is "fixed":
                self.get_data(X, ADV)
                # if self.config.train.restore_iter == 0:
                #     self.get_data(X, ADV)
                # else:
                #     self.load_data_from_file(self.config.train.restore_iter)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

        else:
            if self.config[self.mode].data is "fixed" or X is not None:
                self.get_data(X, ADV)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

    def get_data(self, X=None, ADV=None):
        """Generates data"""
        x_shape = [self.num_instances, self.num_agents, self.num_items]
        adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items]

        if X is None:
            X = self.generate_random_X(x_shape)
        if ADV is None:
            ADV = self.generate_random_ADV(adv_shape)

        self.X = X
        self.ADV = ADV

    def load_data_from_file(self, iter):
        """Loads data from disk"""
        self.X = np.load(os.path.join(self.config.dir_name, "X.npy"))
        self.ADV = np.load(os.path.join(self.config.dir_name, "ADV_" + str(iter) + ".npy"))

    def save_data(self, iter):
        """Saved data to disk"""
        if self.config.save_data is None:
            return

        if iter == 0:
            np.save(os.path.join(self.config.dir_name, "X"), self.X)
        else:
            np.save(os.path.join(self.config.dir_name, "ADV_" + str(iter)), self.ADV)

    def gen_fixed(self):
        i = 0
        if self.mode is "train":
            perm = np.random.permutation(self.num_instances)
        else:
            perm = np.arange(self.num_instances)

        while True:
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.X[idx], self.ADV[:, idx, :, :], idx
            i += 1
            if i * self.batch_size == self.num_instances:
                i = 0
                if self.mode is "train":
                    perm = np.random.permutation(self.num_instances)
                else:
                    perm = np.arange(self.num_instances)

    def gen_online(self):
        x_batch_shape = [self.batch_size, self.num_agents, self.num_items]
        adv_batch_shape = [self.num_misreports, self.batch_size, self.num_agents, self.num_items]
        while True:
            X = self.generate_random_X(x_batch_shape)
            ADV = self.generate_random_ADV(adv_batch_shape)
            yield X, ADV, None

    def update_adv(self, idx, adv_new):
        """Updates ADV for caching"""
        self.ADV[:, idx, :, :] = adv_new

    def generate_random_X(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

    def generate_random_ADV(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

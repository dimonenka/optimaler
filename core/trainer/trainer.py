import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from core.plot import plot


class Trainer(object):
    def __init__(self, configuration, net, clip_op_lambda, device):
        self.net = net
        self.config = configuration
        self.device = device
        self.mode = "train"
        self.clip_op_lambda = clip_op_lambda

        self.writer = SummaryWriter(self.config.save_data)

        self.init_componenents()

    def init_componenents(self):
        self.create_constants()

        self.create_params_to_train()

        self.create_optimizers()

        self.create_masks()

        self.save_config()

    def create_constants(self):
        self.x_shape = dict()
        self.x_shape["train"] = [self.config.train.batch_size, self.config.num_agents, self.config.num_items]
        self.x_shape["val"] = [self.config.val.batch_size, self.config.num_agents, self.config.num_items]

        self.adv_shape = dict()
        self.adv_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.adv_var_shape = dict()
        self.adv_var_shape["train"] = [
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_var_shape["val"] = [
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.u_shape = dict()
        self.u_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
        ]
        self.u_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
        ]

        self.w_rgt = self.config.train.w_rgt_init_val
        self.rgt_target = self.config.train.rgt_target_start
        self.rgt_target_mult = (self.config.train.rgt_target_end / self.config.train.rgt_target_start) ** (
            1.5 / self.config.train.max_iter
        )

    def create_params_to_train(self, train=True, val=True):
        # Trainable variable for find best misreport using gradient by inputs
        self.adv_var = dict()
        if train:
            self.adv_var["train"] = torch.zeros(
                self.adv_var_shape["train"], requires_grad=True, device=self.device
            ).float()
        if val:
            self.adv_var["val"] = torch.zeros(self.adv_var_shape["val"], requires_grad=True, device=self.device).float()

    def create_optimizers(self, train=True, val=True):
        self.opt1 = optim.Adam(self.net.parameters(), self.config.train.learning_rate)

        # Optimizer for best misreport find
        self.opt2 = dict()
        if train:
            self.opt2["train"] = optim.Adam([self.adv_var["train"]], self.config.train.gd_lr)
        if val:
            self.opt2["val"] = optim.Adam([self.adv_var["val"]], self.config.val.gd_lr)

        self.sc_opt2 = dict()
        if train:
            self.sc_opt2["train"] = optim.lr_scheduler.StepLR(self.opt2["train"], 1, self.config.train.gd_lr_step)
        if val:
            self.sc_opt2["val"] = optim.lr_scheduler.StepLR(self.opt2["val"], 1, self.config.val.gd_lr_step)

    def create_masks(self, train=True, val=True):
        self.adv_mask = dict()
        if train:
            self.adv_mask["train"] = np.zeros(self.adv_shape["train"])
            self.adv_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["train"] = tensor(self.adv_mask["train"]).float()

        if val:
            self.adv_mask["val"] = np.zeros(self.adv_shape["val"])
            self.adv_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["val"] = tensor(self.adv_mask["val"]).float()

        self.u_mask = dict()
        if train:
            self.u_mask["train"] = np.zeros(self.u_shape["train"])
            self.u_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["train"] = tensor(self.u_mask["train"]).float()

        if val:
            self.u_mask["val"] = np.zeros(self.u_shape["val"])
            self.u_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["val"] = tensor(self.u_mask["val"]).float()

    def save_config(self):
        print(self.writer.log_dir)
        print(type(self.config))
        with open(self.writer.log_dir + "/config.json", "w") as f:
            json.dump(self.config, f)

    def mis_step(self, x):
        """
        Find best misreport step using gradient by inputs, trainable inputs: self.adv_var variable
        """
        mode = self.mode

        self.opt2[mode].zero_grad()

        # Get misreports
        x_mis, misreports = self.get_misreports_grad(x)

        # Run net for misreports
        a_mis, p_mis = self.net(misreports)

        # Calculate utility
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Calculate loss value
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()

        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()

    def train_op(self, x):
        """
        Loss for main net train
        """
        self.opt1.zero_grad()

        x_mis, misreports = self.get_misreports(x)
        alloc_true, pay_true = self.net(x)
        a_mis, p_mis = self.net(misreports)

        rgt = self.compute_regret(x, alloc_true, pay_true, x_mis, a_mis, p_mis).sum()

        # Revenue
        revenue = self.compute_rev(pay_true)

        # Dual gradient decent
        self.w_rgt = max(
            0,
            self.w_rgt
            + self.config.train.rgt_lr * ((rgt / (revenue + 1e-8)).detach().log().item() - np.log(self.rgt_target)),
        )

        final_loss = -revenue + self.w_rgt * rgt

        # Make a step
        final_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.opt1.step()

        return final_loss, revenue, rgt

    def compute_metrics(self, x):
        """
        Validation metrics
        """
        x_mis, misreports = self.get_misreports_grad(x)

        alloc_true, pay_true = self.net(x)
        a_mis, p_mis = self.net(misreports)

        rgt = self.compute_regret_grad(x, alloc_true, pay_true, x_mis, a_mis, p_mis)

        revenue = self.compute_rev(pay_true)

        return revenue, rgt.mean()

    def compute_rev(self, pay):
        """Given payment (pay), computes revenue
        Input params:
            pay: [num_batches, num_agents]
        Output params:
            revenue: scalar
        """
        return pay.sum(dim=-1).mean()

    def compute_utility(self, x, alloc, pay):
        """Given input valuation (x), payment (pay) and allocation (alloc), computes utility
        Input params:
            x: [num_batches, num_agents, num_items]
            a: [num_batches, num_agents, num_items]
            p: [num_batches, num_agents]
        Output params:
            utility: [num_batches, num_agents]
        """
        return (alloc * x).sum(dim=-1) - pay

    def compute_regret(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        return self.compute_regret_grad(x, a_true, p_true, x_mis, a_mis, p_mis)

    def compute_regret_grad(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        mode = self.mode

        utility = self.compute_utility(x, a_true, p_true)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        utility_true = utility.repeat(self.config.num_agents * self.config[mode].num_misreports, 1)
        excess_from_utility = F.relu(
            (utility_mis - utility_true).view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)
        )

        rgt = excess_from_utility.max(3)[0].max(1)[0].mean(dim=1)
        return rgt

    def get_misreports(self, x):
        return self.get_misreports_grad(x)

    def get_misreports_grad(self, x):
        mode = self.mode
        adv_mask = self.adv_mask[mode].to(self.device)

        adv = self.adv_var[mode].unsqueeze(0).repeat(self.config.num_agents, 1, 1, 1, 1)
        x_mis = x.repeat(self.config.num_agents * self.config[mode].num_misreports, 1, 1)
        x_r = x_mis.view(self.adv_shape[mode])
        y = x_r * (1 - adv_mask) + adv * adv_mask
        misreports = y.view([-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def train(self, generator):
        """
        Main function, full train process
        """
        # Make a generators for train and validation
        self.train_gen, self.val_gen = generator

        iteration = self.config.train.restore_iter

        # Load save model
        if iteration > 0:
            model_path = self.writer.log_dir + "/model_{}".format(iteration)
            state_dict = torch.load(model_path)
            self.net.load_state_dict(state_dict)

        time_elapsed = 0.0

        while iteration < (self.config.train.max_iter):
            tic = time.time()
            self.train_epoch(iteration)

            toc = time.time()
            time_elapsed += toc - tic

            iteration += 1
            self.writer.add_scalar("Train/epoch time", time_elapsed, iteration / 1000)

            if (iteration + 1) % self.config.train.save_iter == 0:
                self.save(iteration + 1)

            # Validation
            if (iteration % self.config.val.print_iter) == 0:
                self.eval(iteration)

    def train_epoch(self, iteration):
        self.mode = "train"
        self.net.train()

        # Get new batch. X - true valuation, ADV - start point for misreport candidates
        # perm - ADV positions in full train dataset
        X, ADV, perm = next(self.train_gen.gen_func)

        x = torch.from_numpy(X).float().to(self.device)

        # Write start adv value for find best misreport variable
        self.adv_var["train"].data = tensor(ADV).float().to(self.device)

        self.misreport_cycle(x)

        # Save found best misreport values in data generator
        if self.config.train.data is "fixed" and self.config.train.adv_reuse:
            self.train_gen.update_adv(perm, self.adv_var["train"].data.cpu())

        # Make a step for net weights updating
        net_loss, train_revenue, train_regret = self.train_op(x)

        self.rgt_target = max(self.rgt_target * self.rgt_target_mult, self.config.train.rgt_target_end)

        if (iteration % self.config.train.print_iter) == 0:
            print("Iteration {}".format(iteration))
            print(
                "Train revenue: {},   regret: {},   net loss: {} , w: {}".format(
                    round(float(train_revenue), 5),
                    round(float(train_regret), 5),
                    round(float(net_loss), 5),
                    round(self.w_rgt, 4),
                )
            )
            self.writer.add_scalar("Train/revenue", train_revenue, iteration / 1000)
            self.writer.add_scalar("Train/regret", train_regret, iteration / 1000)
            self.writer.add_scalar("Train/loss", net_loss, iteration / 1000)
            self.writer.add_scalar("Train/w_rgt", self.w_rgt, iteration / 1000)

    def eval(self, iteration):
        print("Validation on {} iteration".format(iteration))
        self.mode = "val"
        self.net.eval()

        self.eval_grad(iteration)

        if self.config.plot.bool:
            self.plot()

    def eval_grad(self, iteration):
        val_revenue = 0
        val_regret = 0

        for _ in range(self.config.val.num_batches):
            X, ADV, _ = next(self.val_gen.gen_func)
            self.adv_var["val"].data = tensor(ADV).float().to(self.device)

            x = torch.from_numpy(X).float().to(self.device)

            self.misreport_cycle(x)

            val_revenue_batch, val_regret_batch = self.compute_metrics(x)
            val_revenue += val_revenue_batch
            val_regret += val_regret_batch

        val_revenue /= float(self.config.val.num_batches)
        val_regret /= float(self.config.val.num_batches)

        print("Val revenue: {},   regret_grad: {}".format(round(float(val_revenue), 5), round(float(val_regret), 5)))
        self.writer.add_scalar("Validation/revenue", val_revenue, iteration / 1000)
        self.writer.add_scalar("Validation/regret_grad", val_regret, iteration / 1000)

    def plot(self):
        x = np.linspace(self.config.min, self.config.max, self.config.plot.n_points)
        y = x
        if self.config.setting == 'additive_1x2_uniform_416_47':
            y = np.linspace(self.config.min, 7, self.config.plot.n_points)
        x = np.stack([v.flatten() for v in np.meshgrid(x, y)], axis=-1)
        x = np.expand_dims(x, 1)
        x = torch.FloatTensor(x)

        allocation = self.net(x.to(self.device))[0].cpu()
        allocation = (
            allocation.detach()
            .numpy()[:, 0, :]
            .reshape(self.config.plot.n_points, self.config.plot.n_points, self.config.num_items)
        )

        plot(allocation, self.config.save_data, self.config.setting)

    def misreport_cycle(self, x):
        mode = self.mode

        # Find best misreport cycle
        for _ in range(self.config[mode].gd_iter):
            # Make a gradient step, update self.adv_var variable
            self.mis_step(x)

            # Clipping new values of self.adv_var with respect for valuations distribution
            self.clip_op_lambda(self.adv_var[mode])

        for param_group in self.opt2[mode].param_groups:
            param_group["lr"] = self.config[mode].gd_lr

        self.opt2[mode].state = defaultdict(dict)  # reset momentum

    def save(self, iteration):
        torch.save(self.net.state_dict(), self.writer.log_dir + "/model_{}".format(iteration))


class DistillationTrainer(Trainer):
    def __init__(self, configuration, net, target_net, clip_op_lambda, device):
        self.target_net = target_net
        self.mode_target = False
        super().__init__(configuration, net, clip_op_lambda, device)

    def train_epoch(self, iteration):
        self.mode = "train"
        self.net.train()

        # Get new batch. X - true valuation, ADV - start point for misreport candidates
        # perm - ADV positions in full train dataset
        X, ADV, perm = next(self.train_gen.gen_func)

        x = torch.from_numpy(X).float().to(self.device)

        # Write start adv value for find best misreport variable
        self.adv_var["train"].data = tensor(ADV).float().to(self.device)

        if self.config.distill.train_misreports:
            self.misreport_cycle(x)

            # Save found best misreport values in data generator
            if self.config.train.data is "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, self.adv_var["train"].data.cpu())

        # Make a step for net weights updating
        net_loss, train_revenue, target_revenue = self.train_op(x)

        if (iteration % self.config.train.print_iter) == 0:
            print("Iteration {}".format(iteration))
            print(
                "Train revenue: {}, target revenue: {}, net loss: {}".format(
                    round(float(train_revenue), 5),
                    round(float(target_revenue), 5),
                    round(float(net_loss), 5),
                )
            )
            self.writer.add_scalar("Train/revenue", train_revenue, iteration / 1000)
            self.writer.add_scalar("Train/revenue_target", target_revenue, iteration / 1000)
            self.writer.add_scalar("Train/loss", net_loss, iteration / 1000)

    def train_op(self, x):
        """
        Loss for main net train
        """
        self.opt1.zero_grad()

        alloc, final_pay, pay = self.net(x, return_intermediates=True)
        alloc_target, final_pay_target, pay_target = self.target_net(x, return_intermediates=True)
        alloc_target, final_pay_target, pay_target = alloc_target.detach(), final_pay_target.detach(), \
                                                     pay_target.detach()

        # Revenue
        revenue = self.compute_rev(final_pay)
        revenue_target = self.compute_rev(final_pay_target)

        # Minimize KL divergence
        p_log, r_p_log, pt_log, r_pt_log = (pay + 1e-5).log(), (1 - pay + 1e-5).log(), \
                                           (pay_target + 1e-5).log(), (1 - pay_target + 1e-5).log()
        pay_loss = (pay_target * (pt_log - p_log) + (1 - pay_target) * (r_pt_log - r_p_log)).mean()

        a_log, r_a_log, at_log, r_at_log = (alloc + 1e-5).log(), (1 - alloc.sum(dim=1) + 1e-5).log(), \
                                           (alloc_target + 1e-5).log(), (1 - alloc_target.sum(dim=1) + 1e-5).log()
        alloc_loss = (alloc_target * (at_log - a_log)).sum(dim=1).mean()
        alloc_loss += ((1 - alloc_target.sum(dim=1)) * (r_at_log - r_a_log)).mean()
        final_loss = pay_loss + alloc_loss

        if self.config.distill.train_misreports:
            _, misreports = self.get_misreports(x)
            alloc_mis, _, pay_mis = self.net(misreports, return_intermediates=True)
            alloc_mis_target, _, pay_mis_target = self.target_net(misreports, return_intermediates=True)
            alloc_mis_target, pay_mis_target = alloc_mis_target.detach(), pay_mis_target.detach()

            p_log, r_p_log, pt_log, r_pt_log = (pay_mis + 1e-5).log(), (1 - pay_mis + 1e-5).log(), \
                                               (pay_mis_target + 1e-5).log(), (1 - pay_mis_target + 1e-5).log()
            pay_loss = (pay_mis_target * (pt_log - p_log) + (1 - pay_mis_target) * (r_pt_log - r_p_log)).mean()

            a_log, r_a_log, at_log, r_at_log = (alloc_mis + 1e-5).log(), (1 - alloc_mis.sum(dim=1) + 1e-5).log(), \
                                               (alloc_mis_target + 1e-5).log(), (1 - alloc_mis_target.sum(dim=1) + 1e-5).log()
            alloc_loss = (alloc_mis_target * (at_log - a_log)).sum(dim=1).mean()
            alloc_loss += ((1 - alloc_mis_target.sum(dim=1)) * (r_at_log - r_a_log)).mean()
            final_loss += pay_loss + alloc_loss

        # Make a step
        final_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.opt1.step()

        return final_loss, revenue, revenue_target

    def eval_grad(self, iteration):
        val_revenue = 0
        val_regret = 0
        val_revenue_t = 0
        val_regret_t = 0
        val_t_regret = 0
        val_t_regret_t = 0

        for _ in range(self.config.val.num_batches):
            X, ADV, _ = next(self.val_gen.gen_func)
            self.adv_var["val"].data = tensor(ADV).float().to(self.device)

            x = torch.from_numpy(X).float().to(self.device)

            self.misreport_cycle(x)

            val_revenue_batch, val_regret_batch = self.compute_metrics(x)
            val_revenue += val_revenue_batch
            val_regret += val_regret_batch

            self.mode_target = True
            val_revenue_t_batch, val_regret_t_batch = self.compute_metrics(x)
            val_revenue_t += val_revenue_t_batch
            val_regret_t += val_regret_t_batch

            if self.config.distill.validate_target_misreports:
                self.misreport_cycle(x)
                val_t_regret_t += self.compute_metrics(x)[1]
                self.mode_target = False

                val_t_regret += self.compute_metrics(x)[1]

            self.mode_target = False

        val_revenue /= float(self.config.val.num_batches)
        val_regret /= float(self.config.val.num_batches)

        val_revenue_t /= float(self.config.val.num_batches)
        val_regret_t /= float(self.config.val.num_batches)

        print("Val revenue: {}, revenue_target: {}, regret_grad: {}, regret_grad_target: {}".format(
            round(float(val_revenue), 5),
            round(float(val_revenue_t), 5),
            round(float(val_regret), 5),
            round(float(val_regret_t), 5)
        ))

        self.writer.add_scalar("Validation/revenue", val_revenue, iteration / 1000)
        self.writer.add_scalar("Validation/regret_grad", val_regret, iteration / 1000)
        self.writer.add_scalar("Validation/revenue_target", val_revenue_t, iteration / 1000)
        self.writer.add_scalar("Validation/regret_grad_target", val_regret_t, iteration / 1000)

        if self.config.distill.validate_target_misreports:
            val_t_regret /= float(self.config.val.num_batches)
            val_t_regret_t /= float(self.config.val.num_batches)

            print("Val target regret_grad: {}, regret_grad_target: {}".format(
                round(float(val_t_regret), 5),
                round(float(val_t_regret_t), 5)
            ))

            self.writer.add_scalar("Validation_target/regret_grad", val_t_regret, iteration / 1000)
            self.writer.add_scalar("Validation_target/regret_grad_target", val_t_regret_t, iteration / 1000)

    def mis_step(self, x):
        """
        Find best misreport step using gradient by inputs, trainable inputs: self.adv_var variable
        """
        mode = self.mode

        self.opt2[mode].zero_grad()

        # Get misreports
        x_mis, misreports = self.get_misreports_grad(x)

        # Run net for misreports
        if not self.mode_target:
            a_mis, p_mis = self.net(misreports)
        else:
            a_mis, p_mis = self.target_net(misreports)

        # Calculate utility
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Calculate loss value
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()

        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()

    def compute_metrics(self, x):
        """
        Validation metrics
        """
        x_mis, misreports = self.get_misreports_grad(x)

        if not self.mode_target:
            alloc_true, pay_true = self.net(x)
            a_mis, p_mis = self.net(misreports)
        else:
            alloc_true, pay_true = self.target_net(x)
            a_mis, p_mis = self.target_net(misreports)

        rgt = self.compute_regret_grad(x, alloc_true, pay_true, x_mis, a_mis, p_mis)

        revenue = self.compute_rev(pay_true)

        return revenue, rgt.mean()


class TrainerCrossVal(object):
    def __init__(self, configuration, nets, clip_op_lambda, device):
        self.nets = nets
        self.config = configuration
        self.device = device
        self.mode = "val"
        self.clip_op_lambda = clip_op_lambda

        self.writer = SummaryWriter(self.config.save_data)

        self.init_componenents()

    def init_componenents(self):
        self.create_constants()

        self.create_params_to_train(train=False)

        self.create_masks(train=False)

        self.create_optimizers(train=False)

    def create_constants(self):
        self.x_shape = dict()
        self.x_shape["train"] = [self.config.train.batch_size, self.config.num_agents, self.config.num_items]
        self.x_shape["val"] = [self.config.val.batch_size, self.config.num_agents, self.config.num_items]

        self.adv_shape = dict()
        self.adv_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.adv_var_shape = dict()
        self.adv_var_shape["train"] = [
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_var_shape["val"] = [
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.u_shape = dict()
        self.u_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
        ]
        self.u_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
        ]

        self.w_rgt = self.config.train.w_rgt_init_val
        self.rgt_target = self.config.train.rgt_target_start
        self.rgt_target_mult = (self.config.train.rgt_target_end / self.config.train.rgt_target_start) ** (
            1.5 / self.config.train.max_iter
        )

    def create_params_to_train(self, train=True, val=True):
        # Trainable variable for find best misreport using gradient by inputs
        self.adv_var = dict()
        if train:
            self.adv_var["train"] = torch.zeros(
                self.adv_var_shape["train"], requires_grad=True, device=self.device
            ).float()
        if val:
            self.adv_var["val"] = torch.zeros(self.adv_var_shape["val"], requires_grad=True, device=self.device).float()

    def create_optimizers(self, train=True, val=True):
        # Optimizer for best misreport find
        self.opt2 = dict()
        if train:
            self.opt2["train"] = optim.Adam([self.adv_var["train"]], self.config.train.gd_lr)
        if val:
            self.opt2["val"] = optim.Adam([self.adv_var["val"]], self.config.val.gd_lr)

        self.sc_opt2 = dict()
        if train:
            self.sc_opt2["train"] = optim.lr_scheduler.StepLR(self.opt2["train"], 1, self.config.train.gd_lr_step)
        if val:
            self.sc_opt2["val"] = optim.lr_scheduler.StepLR(self.opt2["val"], 1, self.config.val.gd_lr_step)

    def create_masks(self, train=True, val=True):
        self.adv_mask = dict()
        if train:
            self.adv_mask["train"] = np.zeros(self.adv_shape["train"])
            self.adv_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["train"] = tensor(self.adv_mask["train"]).float()

        if val:
            self.adv_mask["val"] = np.zeros(self.adv_shape["val"])
            self.adv_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["val"] = tensor(self.adv_mask["val"]).float()

        self.u_mask = dict()
        if train:
            self.u_mask["train"] = np.zeros(self.u_shape["train"])
            self.u_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["train"] = tensor(self.u_mask["train"]).float()

        if val:
            self.u_mask["val"] = np.zeros(self.u_shape["val"])
            self.u_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["val"] = tensor(self.u_mask["val"]).float()

    def mis_step(self, x):
        """
        Find best misreport step using gradient by inputs, trainable inputs: self.adv_var variable
        """
        mode = self.mode

        self.opt2[mode].zero_grad()

        # Get misreports
        x_mis, misreports = self.get_misreports_grad(x)

        # Run net for misreports
        a_mis, p_mis = self.net(misreports)

        # Calculate utility
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Calculate loss value
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()

        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()

    def compute_metrics(self, x):
        """
        Validation metrics
        """
        x_mis, misreports = self.get_misreports_grad(x)

        alloc_true, pay_true = self.net(x)
        a_mis, p_mis = self.net(misreports)

        rgt = self.compute_regret_grad(x, alloc_true, pay_true, x_mis, a_mis, p_mis)

        revenue = self.compute_rev(pay_true)

        return revenue, rgt.mean()

    def compute_rev(self, pay):
        """Given payment (pay), computes revenue
        Input params:
            pay: [num_batches, num_agents]
        Output params:
            revenue: scalar
        """
        return pay.sum(dim=-1).mean()

    def compute_utility(self, x, alloc, pay):
        """Given input valuation (x), payment (pay) and allocation (alloc), computes utility
        Input params:
            x: [num_batches, num_agents, num_items]
            a: [num_batches, num_agents, num_items]
            p: [num_batches, num_agents]
        Output params:
            utility: [num_batches, num_agents]
        """
        return (alloc * x).sum(dim=-1) - pay

    def compute_regret(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        return self.compute_regret_grad(x, a_true, p_true, x_mis, a_mis, p_mis)

    def compute_regret_grad(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        mode = self.mode

        utility = self.compute_utility(x, a_true, p_true)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        utility_true = utility.repeat(self.config.num_agents * self.config[mode].num_misreports, 1)
        excess_from_utility = F.relu(
            (utility_mis - utility_true).view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)
        )

        rgt = excess_from_utility.max(3)[0].max(1)[0].mean(dim=1)
        return rgt

    def get_misreports(self, x):
        return self.get_misreports_grad(x)

    def get_misreports_grad(self, x):
        mode = self.mode
        adv_mask = self.adv_mask[mode].to(self.device)

        adv = self.adv_var[mode].unsqueeze(0).repeat(self.config.num_agents, 1, 1, 1, 1)
        x_mis = x.repeat(self.config.num_agents * self.config[mode].num_misreports, 1, 1)
        x_r = x_mis.view(self.adv_shape[mode])
        y = x_r * (1 - adv_mask) + adv * adv_mask
        misreports = y.view([-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def train(self, generator):
        """
        Main function, full train process
        """
        # Make a generators for train and validation
        self.train_gen, self.val_gen = generator

        self.eval()

    def eval(self):
        self.mode = "val"
        for net in self.nets.values():
            net.eval()

        self.eval_grad()

    def eval_grad(self):
        metrics = {'RegretNet': {'revenue': 0, 'regret_RegretNet': 0, 'regret_EquivariantNet': 0, 'regret_RegretFormer': 0},
                   'EquivariantNet': {'revenue': 0, 'regret_RegretNet': 0, 'regret_EquivariantNet': 0, 'regret_RegretFormer': 0},
                   'RegretFormer': {'revenue': 0, 'regret_RegretNet': 0, 'regret_EquivariantNet': 0, 'regret_RegretFormer': 0}}

        for _ in range(self.config.val.num_batches):
            X, ADV, _ = next(self.val_gen.gen_func)
            self.adv_var["val"].data = tensor(ADV).float().to(self.device)

            x = torch.from_numpy(X).float().to(self.device)

            for name_misreport, net in self.nets.items():
                self.net = net
                self.misreport_cycle(x)

                for name_regret, net in self.nets.items():
                    self.net = net
                    rev, reg = self.compute_metrics(x)
                    if name_misreport == name_regret:
                        metrics[name_regret]['revenue'] += rev
                    metrics[name_regret]['regret_' + name_misreport] += reg

        for name in metrics.keys():
            for met in metrics[name].keys():
                metrics[name][met] /= self.config.val.num_batches

        f = lambda z: round(float(z), 6)
        print(f"""
                            revenue    regret_on_RegretNet    regret_on_EquivariantNet    regret_on_RegretFormer
        RegretNet          {f(metrics['RegretNet']['revenue'])}   {f(metrics['RegretNet']['regret_RegretNet'])}              {f(metrics['RegretNet']['regret_EquivariantNet'])}                  {f(metrics['RegretNet']['regret_RegretFormer'])}
        EquivariantNet     {f(metrics['EquivariantNet']['revenue'])}   {f(metrics['EquivariantNet']['regret_RegretNet'])}              {f(metrics['EquivariantNet']['regret_EquivariantNet'])}                  {f(metrics['EquivariantNet']['regret_RegretFormer'])}
        RegretFormer       {f(metrics['RegretFormer']['revenue'])}   {f(metrics['RegretFormer']['regret_RegretNet'])}              {f(metrics['RegretFormer']['regret_EquivariantNet'])}                  {f(metrics['RegretFormer']['regret_RegretFormer'])}
        """)

    def misreport_cycle(self, x):
        mode = self.mode

        # Find best misreport cycle
        for _ in range(self.config[mode].gd_iter):
            # Make a gradient step, update self.adv_var variable
            self.mis_step(x)

            # Clipping new values of self.adv_var with respect for valuations distribution
            self.adv_var[mode].data.clamp_(self.config.min, self.config.max)

        for param_group in self.opt2[mode].param_groups:
            param_group["lr"] = self.config[mode].gd_lr

        self.opt2[mode].state = defaultdict(dict)  # reset momentum

import random

import torch
import torch.nn.functional as F

from utility.efficient_allocation import oracle


def gen_valuations_normal(size: object, mean_of_mean: object = 10, std_of_mean: object = 1, mean_of_std: object = 2, std_of_std: object = 0.5) -> object:
    """
    :return: tensor of valuations shaped as (size,)
    """
    data = torch.zeros(size).float()
    mean = torch.distributions.Normal(mean_of_mean, std_of_mean).sample((size,))
    std = F.relu(torch.distributions.Normal(mean_of_std - 1e-3, std_of_std).sample((size,))) + 1e-3
    for i in range(size):
        data[i] = torch.distributions.Normal(mean[i] - 1e-3, std[i]).sample()
    return F.relu(data) + 1e-3


def gen_auctions_normal(
    size=100000,
    n_participants=10,
    n_items=8,
    language="marginal",
    mean_of_mean=10,
    std_of_mean=1,
    mean_of_std=2,
    std_of_std=0.5,
):
    """
    :param language: bidding language, either
        'additive' for heterogenous goods and additive utilities
        'marginal' for homogenous goods and marginally decreasing utilities
            (utility of a bundle is sum of item-wise utilities, s.t. each new item produces less utility) or
        'unit-demand' for heterogenous goods and unit demand (utility of a bundle is max of item-wise utilities)
        * 'hierarchical' for hierarchical bundles?
    :return: auction tensor shaped as (size, n_participants, n_items)
    """

    data = gen_valuations_normal(size * n_participants * n_items, mean_of_mean, std_of_mean, mean_of_std, std_of_std)
    data = data.view(size, n_participants, n_items)
    data = prepare_auctions(data, language)
    return data


def prepare_auctions(auctions, language="marginal"):
    if language == "additive":
        pass
    elif language == "marginal":
        auctions = torch.sort(auctions, dim=2, descending=True)[0]
        auctions = torch.cumsum(
            auctions, 2
        )  ### this is different from the paper, but the marginal utilities can increase otherwise
    elif language == "unit-demand":
        pass
    else:
        raise NotImplementedError("Only 'additive', 'marginal', and 'unit-demand' languages are implemented")
    return auctions


def get_batch(auctions, batch_size=256, return_idx=False):
    batch_size = min(batch_size, len(auctions))
    # idx = torch.randperm(len(auctions))[:batch_size]
    idx = torch.Tensor(random.sample(range(len(auctions)), batch_size)).long()
    if not return_idx:
        return auctions[idx]
    else:
        return idx, auctions[idx]


def get_representation(data, language="marginal"):
    """
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :return: auction representations that are used as input for DeepMindNet, tensor shaped as (batch_size, n_agents, n_items, n_channels)
    """
    old_shape = list(data.shape)
    n_items = old_shape[2]
    representation = [data]

    for i in range(n_items):
        allocation = oracle(data, language, i + 1).float()
        utility = data * allocation
        representation.extend([allocation, utility])

    representation = torch.stack(representation, -1)
    return representation


if __name__ == "__main__":
    auctions = gen_auctions_normal(5, n_participants=2, n_items=3, language="unit-demand")
    print(auctions.size())

    batch = get_batch(auctions, 5)
    print(batch.size())
    print(batch)

    cube = get_representation(batch, "unit-demand")
    print(batch.shape, cube.shape)
    print(cube)

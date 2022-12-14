import torch
import torch.nn.functional as F
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix


def oracle(data, language="marginal", n_best_items=None):
    """
    Allocation oracle.
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :param language: bidding language, either
        'additive' for heterogenous goods and additive utilities,
        'marginal' for homogenous goods and marginally decreasing utilities
            (utility of a bundle is sum of item-wise utilities, s.t. each new item produces less utility) or
        'unit-demand' for heterogenous goods and unit demand (utility of a bundle is max of item-wise utilities)
        * 'hierarchical' for hierarchical bundles?
    :param n_best_items: number of items to allocate, the default is all items
    :return: allocation, i.e. binary tensor with same shape as valuation
    """

    size, n_participants, n_items = data.size()
    if n_best_items is None:
        n_best_items = n_items
    n_best_items = min(n_best_items, n_items)

    if language == "additive":
        allocation = torch.argmax(data, 1, True)
        allocation = F.one_hot(allocation)[:, 0].permute((0, 2, 1))
        allocation = topk_allocation(data, allocation, n_best_items)

    elif language == "marginal":
        allocation = torch.zeros((size, n_participants)).long()
        for i, auction in enumerate(data):
            auction_copy = auction.clone()
            v, idx = auction_copy[:, 0], torch.zeros((n_participants,)).long()
            for j in range(n_best_items):
                part = torch.argmax(v).item()
                idx[part] += 1
                if j != n_items - 1:
                    v[part] = auction_copy[part, idx[part]] - auction_copy[part, idx[part] - 1]
            allocation[i] = idx
        allocation = torch.zeros(size, n_participants, n_items + 1).scatter_(2, allocation.unsqueeze(-1), 1)[:, :, 1:]

    elif language == "unit-demand":
        n_best_items = min(n_best_items, n_participants)
        allocation = torch.zeros(size, n_participants, n_items)
        for i, auction in enumerate(data):
            allocation_cur = torch.zeros(n_participants, n_items)
            graph = csr_matrix(-auction.detach().numpy())
            graph = bipartite.from_biadjacency_matrix(graph)
            matching = bipartite.matching.minimum_weight_full_matching(graph)
            for part in range(n_participants):
                if part in matching.keys():
                    item = matching[part] - n_participants
                    allocation_cur[part, item] = 1
            allocation[i] = allocation_cur
        allocation = topk_allocation(data, allocation, n_best_items)

    else:
        raise NotImplementedError("Only 'additive', 'marginal', and 'unit-demand' languages are implemented")

    return allocation


def topk_allocation(data, allocation, n_best_items):
    threshold = torch.topk((data * allocation).reshape(data.shape[0], -1), n_best_items, -1)[0][:, -1].view(-1, 1, 1)
    allocation[data < threshold] = 0
    return allocation


def get_v(data, allocation):
    """
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :param allocation: efficient allocation, output of oracle, binary tensor shaped as (batch_size, n_agents, n_items)

    :return: total value of objects gained by each agent in each auction with respect to the efficient allocations,
        tensor shaped as (batch_size, n_agents)
    """
    return torch.sum(data * allocation, 2)


def delete_agent(x, i):
    mask = torch.ones(x.size()[1]).int()
    mask[i] = 0
    return x[:, mask.bool()]


def get_v_sum_but_i(v):
    return torch.cat([torch.sum(delete_agent(v, i), dim=1).view(-1, 1) for i in range(v.size()[1])], dim=1)


if __name__ == "__main__":
    from utility.data_generation import prepare_auctions

    batch = torch.FloatTensor(
        [
            [[1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [1, 2, 3]],
            [[2, 3, 4], [1, 2, 3]],
            [[2, 2, 2], [1, 2, 3]],
            [[2, 2, 2], [1, 3, 3]],
            [[5, 6, 7], [1, 2, 3]],
            [[3, 2, 1], [1, 2, 3]],
            [[1, 3, 5], [1, 2, 3]],
        ]
    )

    print(batch)
    print(oracle(batch, "additive"))

    print(batch)
    print(oracle(batch, "unit-demand"))

    batch = prepare_auctions(batch, "marginal")
    print(batch)
    print(oracle(batch, "marginal"))

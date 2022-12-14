import torch

from utility.efficient_allocation import delete_agent, get_v, get_v_sum_but_i, oracle


def VCG(batch, language="marginal"):
    """
    VCG efficient and truthful mechanism.
    :param batch: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :return: VCG prices t, tensor shaped as (batch_size, n_agents)
    """
    allocation = oracle(batch, language)
    v = get_v(batch, allocation)
    v_sum_but_i = get_v_sum_but_i(v)

    h = []
    for i in range(batch.shape[1]):
        batch_cur = delete_agent(batch, i)
        allocation_cur = oracle(batch_cur, language)
        v_cur = get_v(batch_cur, allocation_cur)
        v_sum_cur = v_cur.sum(dim=-1).view(-1, 1)
        h.append(v_sum_cur)
    h = torch.cat(h, dim=1)

    t = h - v_sum_but_i
    return t


if __name__ == "__main__":
    n_auctions, n_agents, n_items = 10000000, 2, 2

    batch = torch.rand((n_auctions, n_agents, n_items))
    prices = VCG(batch, 'additive').numpy()
    print("payoff", float(prices.sum()) / n_auctions)

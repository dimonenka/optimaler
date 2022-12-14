import numpy as np
import torch
from scipy.optimize import newton
from scipy.stats import expon, gamma, truncnorm

from utility.distributions import IrwinHall


def get_r_star(pdf, cdf, x0=0.0, tol=1.48e-08, maxiter=100, **kwargs):
    fun = lambda x: x - (1 - cdf(x)) / pdf(x)
    r_star = newton(fun, x0=x0, tol=tol, maxiter=maxiter)
    return r_star


def get_r_star_normal(loc=0, std=1, **kwargs):
    dist = truncnorm(loc=loc, scale=std)
    return get_r_star(dist.pdf, dist.cdf, x0=loc, **kwargs)


def get_r_star_expon(loc=0, std=1, **kwargs):
    dist = expon(loc=loc, scale=std)
    return get_r_star(dist.pdf, dist.cdf, x0=loc, **kwargs)


def get_r_star_gamma(n=1, loc=0, std=1, **kwargs):
    dist = gamma(a=n, loc=loc, scale=std)
    return get_r_star(dist.pdf, dist.cdf, x0=n + loc, **kwargs)


def get_r_star_uniform(b=1, **kwargs):
    return 0.5 * b


def get_r_star_irwin_hall(n=1, a=0, b=1, **kwargs):
    dist = IrwinHall(n=n, a=a, b=b)
    return get_r_star(dist.pdf, dist.cdf, x0=b / 2, **kwargs)


def second_price(batch, reserve=None):
    """
    Second-price single-good auction with reserve price.
    :param batch: tensor with bids (valuations) in single-item auctions shaped as (batch_size, n_agents)
    :param reserve: reserve price, i.e. the item is not allocated if all bids are below it;
        effectively, reserve price acts as an additional dummy bidder
    :return: tuple (allocation, prices), where both allocation and prices are tensors shaped as (batch_size, n_agents)
    """

    allocation = torch.zeros_like(batch).int()
    idx = torch.argmax(batch, dim=1)
    allocation.scatter_(1, idx.view(-1, 1), 1)

    if batch.shape[1] > 1:
        top2 = torch.topk(batch, 2, dim=1)[0][:, 1]
    else:
        top2 = torch.zeros((batch.shape[0],))
    if reserve is not None:
        top2[top2 < reserve] = reserve

    prices = torch.zeros_like(batch)
    prices.scatter_(1, idx.view(-1, 1), top2.view(-1, 1))

    if reserve is not None:
        mask = batch < reserve
        allocation[mask] = 0
        prices[mask] = 0

    return allocation, prices


def Myerson(batch, reserve=None, dist="uniform", **dist_kwargs):
    """
    Myerson optimal and truthful mechanism for single-item auctions with common valuation distribution.
    It is equivalent to the second-price auction with a specific reserve price.
    :param batch: tensor with bids (valuations) in single-item auctions shaped as (batch_size, n_agents)
    :param reserve: reserve price; if None, reserve is computed with respect to dist and dist_kwargs
    :param dist: 'uniform', 'normal' (truncated normal), or 'expon' (exponential), 'gamma', 'irwin-hall', or 'custom'
    :param dist_kwargs:
        for 'uniform':
            b: right border of the distribution
        for 'normal' and 'expon':
            loc, std: mean and standard deviation parameters of the distribution
            tol, maxiter: parameters of newton optimizer
        for 'gamma': same as 'normal' and 'expon', plus
            n: shape of the distribution
        for 'irwin-hall':
            a, b: left and right border of single uniform distribution
            n: number of summed uniform random variables
        for 'custom':
            pdf, cdf: probability and cumulative density functions of the distribution
            x0, tol, maxiter: parameters of newton optimizer
    :return: tuple (allocation, prices), where both allocation and prices are tensors shaped as (batch_size, n_agents)
    """

    if reserve is None:
        if dist == "uniform":
            reserve = get_r_star_uniform(**dist_kwargs)
        elif dist == "normal":
            reserve = get_r_star_normal(**dist_kwargs)
        elif dist == "expon":
            reserve = get_r_star_expon(**dist_kwargs)
        elif dist == "gamma":
            reserve = get_r_star_gamma(**dist_kwargs)
        elif dist == "irwin-hall":
            reserve = get_r_star_irwin_hall(**dist_kwargs)
        elif dist == "custom":
            reserve = get_r_star(**dist_kwargs)
        else:
            raise NotImplementedError(
                'Parameter dist is either "uniform", "normal", "expon", "gamma", "irwin-hall", or "custom"'
            )

    return second_price(batch, reserve)


def Myerson_item_wise(batch, reserve=None, dist="uniform", **dist_kwargs):
    """
    Separate Myerson auctions for each item.
    :param batch: tensor with bids (valuations) shaped as (batch_size, n_agents, n_items)
    :return: tuple (allocation, prices), where both allocation and prices are tensors shaped as (batch_size, n_agents, n_items)
    """
    shape_init = batch.shape
    batch = batch.permute(1, 0, 2).reshape(shape_init[1], -1).permute(1, 0)

    allocation, prices = Myerson(batch, reserve=reserve, dist=dist, **dist_kwargs)
    allocation = allocation.permute(1, 0).reshape(shape_init[1], shape_init[0], shape_init[2]).permute(1, 0, 2)
    prices = prices.permute(1, 0).reshape(shape_init[1], shape_init[0], shape_init[2]).permute(1, 0, 2)
    return allocation, prices


def Myerson_bundled(batch, reserve=None, dist="uniform", **dist_kwargs):
    """
    Myerson auction for the entire set of items sold as one unit.
    Distribution parameters dist_kwargs are changed accordingly if dist is 'uniform', 'normal', or 'expon'.
    Otherwise, set dist to custom and change pdf and cdf in dist_kwargs manually to account for the summation of valuations.
    :param batch: tensor with bids (valuations) shaped as (batch_size, n_agents, n_items)
    :return: tuple (allocation, prices), where both allocation and prices are tensors shaped as (batch_size, n_agents, n_items)
    """
    shape_init = batch.shape
    n_items = shape_init[-1]

    if dist == "uniform":
        # sum of n independent uniform random variables has Irwin-Hall distribution
        dist = "irwin-hall"
        dist_kwargs["n"] = n_items
    elif dist == "normal":
        # sum of n independent normal random variables N(mean, var) has normal distribution N(n * mean, n * var)
        dist_kwargs["loc"] = dist_kwargs["loc"] * n_items
        dist_kwargs["std"] = dist_kwargs["std"] * np.sqrt(n_items)
    elif dist == "expon":
        # sum of n independent exponential random variables exp(lambda) has gamma distribution gamma(n, lambda)
        dist = "gamma"
        dist_kwargs["n"] = n_items
    else:
        raise Warning("Be sure to change pdf and cdf in dist_kwargs to account for the summation of valuations")

    allocation, prices = Myerson(batch.sum(dim=-1), reserve=reserve, dist=dist, **dist_kwargs)
    allocation = allocation.unsqueeze(-1).expand(shape_init)
    prices = prices.unsqueeze(-1).expand(shape_init) / n_items
    return allocation, prices


if __name__ == "__main__":
    ###  theoretical payoffs below are for the auctions with 2 participants
    n_auctions, n_agents, n_items = 10000000, 2, 2

    batch = torch.rand((n_auctions, n_agents))
    allocation, prices = Myerson(batch)
    print("single item", "\nempirical payoff", float(prices.sum()) / n_auctions, "\ntheoretical payoff", 5 / 12)

    batch = torch.rand((n_auctions, n_agents, n_items))
    allocation, prices = Myerson_item_wise(batch)
    print("item-wise", "\nempirical payoff", float(prices.sum()) / n_auctions, "\ntheoretical payoff", 5 / 12 * n_items)

    allocation, prices = Myerson_bundled(batch)
    print("bundled", "\nempirical payoff", float(prices.sum()) / n_auctions)

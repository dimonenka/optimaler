import numpy as np
import scipy.special as sc
from scipy.stats import rv_continuous

"""
Based on https://github.com/rlucas7/scipy/blob/irwinhall/scipy/stats/_continuous_distns.py
"""


class IrwinHall(rv_continuous):
    """
    Irwin-hall is the distribution of sum of identical uniform random variables.
    """

    def __init__(self, n=1, a=0, b=1):
        """
        :param n: number of summed uniform random variables
        :param a, b: left and right border of single uniform distribution
        """
        super().__init__()
        self.n, self.a, self.b = n, a, b
        self.loc, self.scale = a, b - a
        self.left, self.right = self.loc, self.loc + self.scale * self.n

    def pdf(self, x):
        n = self.n
        if not self.left <= x <= self.right:
            return 0
        x = (x - self.loc) / self.scale

        fl_x = int(np.floor(x)) + 1
        kernel_list = (np.arange(fl_x) - x) ** (n - 1) * sc.binom(n, np.arange(fl_x))
        kernel_list[1::2] *= -1
        return abs(kernel_list.sum()) / sc.factorial(n - 1) / self.scale

    def cdf(self, x):
        n = self.n
        if x <= self.left:
            return 0
        if x >= self.right:
            return 1
        x = (x - self.loc) / self.scale

        fl_x = int(np.floor(x)) + 1
        kernel_list = (np.arange(fl_x) - x) ** n * sc.binom(n, np.arange(fl_x))
        kernel_list[1::2] *= -1
        return abs(kernel_list.sum()) / sc.factorial(n)


if __name__ == "__main__":
    dist = IrwinHall(n=1, a=0, b=1)
    lst = [-1, 0, 0.3, 0.5, 1, 1.5, 2, 2.5, 3]
    print([dist.pdf(x) for x in lst])
    print([dist.cdf(x) for x in lst])

import os

import matplotlib.pyplot as plt
import numpy as np


def plot(alloc, dir_name, setting):
    if setting == "additive_1x2_uniform":
        plot_a_1x2_u(alloc, dir_name)
    elif setting == "additive_1x2_uniform_416_47":
        plot_a_1x2_u_416_47(alloc, dir_name)


def plot_a_1x2_u(alloc, dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # 1st item
    x1 = (2.0 - np.sqrt(2.0)) / 3.0
    x2 = 2.0 / 3.0
    points = [(x1, 1.0), (x1, x2), (x2, x1), (x2, 0)]
    x = list(map(lambda x: x[0], points))
    y = list(map(lambda x: x[1], points))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    plt.plot(x, y, linewidth=2, linestyle="--", c="black")
    img = ax.imshow(alloc[::-1, :, 0], extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, cmap="YlOrRd")

    plt.text(0.25, 0.25, s="0", color="black", fontsize="10", fontweight="bold")
    plt.text(0.65, 0.65, s="1", color="black", fontsize="10", fontweight="bold")

    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    plt.title("Prob. of allocating item 1")
    _ = plt.colorbar(img, fraction=0.046, pad=0.04)

    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(dir_name, "alloc1.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)

    # 2nd item
    x1 = (2.0 - np.sqrt(2.0)) / 3.0
    x2 = 2.0 / 3.0
    points = [(0.0, x2), (x1, x2), (x2, x1), (1.0, x1)]

    x = list(map(lambda x: x[0], points))
    y = list(map(lambda x: x[1], points))

    plt.rcParams.update({"font.size": 10, "axes.labelsize": "x-large"})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    plt.plot(x, y, linewidth=2, linestyle="--", c="black")
    img = ax.imshow(alloc[::-1, :, 1], extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, cmap="YlOrRd")

    plt.text(0.25, 0.25, s="0", color="black", fontsize="10", fontweight="bold")
    plt.text(0.65, 0.65, s="1", color="black", fontsize="10", fontweight="bold")

    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    plt.title("Prob. of allocating item 2")
    _ = plt.colorbar(img, fraction=0.046, pad=0.04)

    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(dir_name, "alloc2.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)


def plot_a_1x2_u_416_47(alloc, dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    points = [(4, 6), (8, 4), (8, 7)]
    x = list(map(lambda x: x[0], points))
    y = list(map(lambda x: x[1], points))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    plt.plot(x, y, linewidth=2, linestyle='--', c='black')
    img = ax.imshow(alloc[::-1, :, 0], extent=[4, 16, 4, 7], vmin=0.0, vmax=1.0, cmap='YlOrRd', aspect=4)

    plt.text(5, 4.5, s='0', color='black', fontsize='10', fontweight='bold')
    plt.text(5.25, 6, s='0.5', color='black', fontsize='10', fontweight='bold')
    plt.text(11.5, 5.5, s='1', color='black', fontsize='10', fontweight='bold')

    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    plt.title("Prob. of allocating item 1")
    _ = plt.colorbar(img, fraction=0.046, pad=0.04)

    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(dir_name, "alloc1.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)

    # 2nd item

    points = [(4, 6), (8, 4)]

    x = list(map(lambda x: x[0], points))
    y = list(map(lambda x: x[1], points))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    plt.plot(x, y, linewidth=2, linestyle='--', c='black')
    img = ax.imshow(alloc[::-1, :, 1], extent=[4, 16, 4, 7], vmin=0.0, vmax=1.0, cmap='YlOrRd', aspect=4)

    plt.text(5, 4.5, s='0', color='black', fontsize='10', fontweight='bold')
    plt.text(11.5, 5.5, s='1', color='black', fontsize='10', fontweight='bold')

    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    plt.title("Prob. of allocating item 2")
    _ = plt.colorbar(img, fraction=0.046, pad=0.04)

    fig.set_size_inches(4, 3)
    plt.savefig(os.path.join(dir_name, "alloc2.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)

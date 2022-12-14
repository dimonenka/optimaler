import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib

from plots.colors import Neon


def plot_with_intervals(min_data, max_data, mean_data, \
    inds=None, label='RegretNet', y_lim=None, x_lim=200, c=Neon.RED.norm, lw=3, linestyle='-', dashes=None, ax=None):

    ax = sns.lineplot(x=inds, y=mean_data, color=c, linewidth=lw, label=label, linestyle=linestyle, dashes=dashes, ci=95, ax=ax)
    ax.fill_between(inds, y1=min_data, y2=max_data, alpha=0.1, color=c)
    ax.lines[-1].set_linestyle(linestyle)
    ax.legend(handlelength=3)
    # ax.set_ylim(0, y_lim)
    ax.set_xlim(0, x_lim)
    if dashes is not None:
        ax.lines[-1].set_dashes(dashes)
    return ax


def style_ax(ax, ac, tc, move_right=False, dashed=False):
    ax.title.set_color(ac)
    ax.xaxis.label.set_color(ac)
    ax.yaxis.label.set_color(ac)
    if move_right:
        ax.spines['left'].set_position(('axes', 1.015))
    if dashed:
        ax.spines['left'].set_linestyle((1, (1, 2)))
    for spine in ax.spines.values():
        spine.set_color(tc)
        spine.set_linewidth(2)


def fig():
    fig = plt.gcf()
    fig.set_size_inches(7, 5, forward=True)


def save(fPath):
    fig = plt.gcf()
    fig.savefig(fPath, dpi=200, bbox_inches='tight', pad_inches=0)


def process_axes(axes, log=False, ylim=None):
    font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    for num, ax in enumerate(axes):
        ax.yaxis.reset_ticks()
        ax.xaxis.set_ticks([0, 50, 100, 150, 200])
        if log:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(*ylim)

        dashed = num == 0 and len(axes) > 1
        # ----- ticks -----
        # ticks_ax(ax, 18, Neon.BLACK.norm, pos_right)
        ts = 20
        tc = Neon.BLACK.norm
        pad = 5

        ax.tick_params(axis='y', labelsize=ts, labelcolor=tc, direction='out', pad=pad)
        ax.tick_params(axis='x', labelsize=ts, labelcolor=tc)
        labels = ax.xaxis.get_majorticklabels()#  + ax.yaxis.get_majorticklabels() 
        for label in labels:
            label.set_fontweight('bold')

        # ----- style ------
        style_ax(ax, Neon.BLACK.norm, Neon.BLACK.norm, False, dashed)
    fig()

import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
from plots.colors import Neon
from plots import plot_config


def plot_from_csv():
    for tag in TAGS:
        PIC_DIR = PATH + 'pics/' + tag[:-4] + '/'
        if not os.path.exists(PIC_DIR): os.mkdir(PIC_DIR)
        for k in RGT_LEVELS.keys():
            if not os.path.exists(PIC_DIR + k): os.mkdir(PIC_DIR + k)

        df = pd.read_csv(PATH + tag)
        for setting in df['setting'].unique():
            df_ = df[df['setting'] == setting].drop(columns=['seed', 'setting'])
            df_mean = df_.groupby('network').mean().T
            df_min = df_.groupby('network').min().T
            df_max = df_.groupby('network').max().T

            ylim_rev = None
            if setting.split('_')[2] == '1x2':
                ylim_rev = (0.45, 0.65)
            elif setting.split('_')[2] == '2x2':
                ylim_rev = (0.7, 1)
            elif setting.split('_')[2] == '2x3':
                ylim_rev = (1, 1.52)
            elif setting.split('_')[2] == '2x5':
                ylim_rev = (1.8, 2.7)
            elif setting.split('_')[2] == '3x10':
                ylim_rev = (4, 6.5)

            for k, v in RGT_LEVELS.items():
                ax = None
                for network in v:
                    network_name = network.split('_')[0]
                    ax = plot_config.plot_with_intervals(
                        df_min[network].values,
                        df_max[network].values,
                        df_mean[network].values,
                        inds=df_mean.index.astype(int),
                        label=network_name,
                        y_lim=None,
                        x_lim=None,
                        c=COLORS[network_name],
                        lw=3,
                        linestyle=LINESTYLES[network_name],
                        ax=ax)

                ax.legend([], [], frameon=False)
                ylim = None
                if tag.endswith('regret_grad.csv'):
                    ylim = (10 ** -5, 0.1)
                elif tag.endswith('w_rgt.csv'):
                    ylim = (1, 2*10**4)
                elif tag.endswith('revenue.csv'):
                    ylim = ylim_rev
                plot_config.process_axes([ax], log=not tag.endswith('revenue.csv'), ylim=ylim)
                plot_config.save(PIC_DIR + k + setting)
                ax.clear()


def legend_from_csv():
    colors = [[k, v] for k, v in COLORS.items()]
    f = lambda m, c: plt.plot([], [], color=c, ls=m)[0]
    handles = [f(LINESTYLES[colors[i][0]], colors[i][1]) for i in range(3)]
    labels = [NAMES[colors[i][0]] for i in range(3)]
    legend = plt.legend(handles, labels, ncol=3, loc=3, framealpha=1, frameon=True)

    def export_legend(legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        legend.get_window_extent()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=500, bbox_inches=bbox)

    plt.grid(False)
    export_legend(legend, PATH + 'legend.png')
    plt.grid(True)


def table_from_csv():
    df_rev = pd.read_csv(PATH + TAGS['revenue'])
    df_rev = df_rev.groupby(['setting', 'network', 'seed'])['200'].mean().sort_index()
    df_rev = df_rev.round(3)

    df_reg = pd.read_csv(PATH + TAGS['regret'])
    df_reg = df_reg.groupby(['setting', 'network', 'seed'])['200'].mean().sort_index()
    df_reg = df_reg.round(5)

    df = df_rev.rename('revenue').to_frame()
    df['regret'] = df_reg
    df.reset_index(inplace=True)
    df['R_max'] = '1e-3'
    df.loc[df['network'].isin(['standard_tight', 'exchangeable_tight', 'attention_tight']), 'R_max'] = '1e-4'
    df['network'] = df['network'].apply(lambda x: x.split('_')[0])
    df.loc[df['network'] == 'standard', 'network'] = 'RegretNet'
    df.loc[df['network'] == 'exchangeable', 'network'] = 'EquivariantNet'
    df.loc[df['network'] == 'attention', 'network'] = 'RegretFormer'
    df['setting'] = df['setting'].apply(lambda x: x.split('_')[2])

    # aggregated std
    print(df.groupby(['R_max', 'setting', 'network'])['revenue'].std().dropna().reset_index().groupby('setting').mean())
    print(df.groupby(['R_max', 'setting', 'network'])['regret'].std().dropna().reset_index().groupby('R_max').mean())

    df = df.groupby(['R_max', 'setting', 'network']).mean()
    df = df.pivot_table(index=['R_max', 'setting'], columns='network', values=['revenue', 'regret'])
    df = df[[['revenue', 'RegretNet'], ['regret', 'RegretNet'],
             ['revenue', 'EquivariantNet'], ['regret', 'EquivariantNet'],
             ['revenue', 'RegretFormer'], ['regret', 'RegretFormer']
             ]]
    df.columns = df.columns.swaplevel(0, 1)
    print(df)
    df.to_csv(PATH + 'df.csv')


if __name__ == '__main__':
    PATH = 'runs/ready/csv/'
    TAGS = ['Train_w_rgt.csv', 'Validation_revenue.csv', 'Validation_regret_grad.csv']
    RGT_LEVELS = {'normal/': ['standard', 'exchangeable', 'attention'],
                  'tight/': ['standard_tight', 'exchangeable_tight', 'attention_tight']}
    COLORS = {'standard': Neon.CYAN.norm, 'exchangeable': Neon.GREEN.norm, 'attention': Neon.RED.norm}
    LINESTYLES = {'standard': '--', 'exchangeable': '-.', 'attention': '-'}
    NAMES = {'standard': 'RegretNet', 'exchangeable': 'EquivariantNet', 'attention': 'RegretFormer'}

    legend_from_csv()

    sns.set_theme()
    plot_from_csv()

    TAGS = {'revenue': 'Validation_revenue.csv', 'regret': 'Validation_regret_grad.csv'}

    table_from_csv()

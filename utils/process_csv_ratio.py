import pandas as pd
from plots.colors import Neon


def table_from_csv():
    cols = [str(x) for x in range(190, 201, 5)]

    df_rev = pd.read_csv(PATH + TAGS['revenue'])
    df_rev = df_rev.groupby(['setting', 'network'])['200'].mean().sort_index()
    df_rev = df_rev.round(3)

    df_reg = pd.read_csv(PATH + TAGS['regret'])
    df_reg = df_reg.groupby(['setting', 'network'])['200'].mean().sort_index()
    df_reg = df_reg.round(5)

    df_revt = pd.read_csv(PATH + TAGS['revenue_train'])
    df_revt = df_revt.groupby(['setting', 'network'])[cols].mean().mean(1).sort_index()
    df_revt = df_revt.round(3)

    df_regt = pd.read_csv(PATH + TAGS['regret_train'])
    df_regt = df_regt.groupby(['setting', 'network'])[cols].mean().mean(1).sort_index()
    df_regt = df_regt.round(5)

    df = df_rev.rename('revenue').to_frame()
    df['revenue_train'] = df_revt
    df['regret'] = df_reg
    df['regret_train'] = df_regt
    df.reset_index(inplace=True)
    df['R_max'] = '1e-3'
    df.loc[df['network'].isin(['standard_tight', 'exchangeable_tight', 'attention_tight']), 'R_max'] = '1e-4'
    df['network'] = df['network'].apply(lambda x: x.split('_')[0])
    df.loc[df['network'] == 'standard', 'network'] = 'RegretNet'
    df.loc[df['network'] == 'exchangeable', 'network'] = 'EquivariantNet'
    df.loc[df['network'] == 'attention', 'network'] = 'RegretFormer'
    df['setting'] = df['setting'].apply(lambda x: x.split('_')[2])

    df['regret_train_ratio'] = df['regret_train'] / df['revenue_train'] / df['R_max'].astype(float)
    df['regret_val_ratio'] = df['regret'] / df['revenue'] / df['R_max'].astype(float) * df['setting'].apply(lambda x: int(x[0]))

    df = df.pivot_table(index=['R_max', 'setting'], columns='network', values=['regret_train_ratio', 'regret_val_ratio'])
    df = df[[['regret_train_ratio', 'RegretNet'], ['regret_val_ratio', 'RegretNet'],
             ['regret_train_ratio', 'EquivariantNet'], ['regret_val_ratio', 'EquivariantNet'],
             ['regret_train_ratio', 'RegretFormer'], ['regret_val_ratio', 'RegretFormer']
             ]]
    df.columns = df.columns.swaplevel(0, 1)
    df = df.round(2)
    print(df)
    df.to_csv(PATH + 'df_regret_ratio.csv')


if __name__ == '__main__':
    PATH = 'runs/ready/csv/'
    TAGS = ['Train_w_rgt.csv', 'Validation_revenue.csv', 'Validation_regret_grad.csv']
    RGT_LEVELS = {'normal/': ['standard', 'exchangeable', 'attention'],
                  'tight/': ['standard_tight', 'exchangeable_tight', 'attention_tight']}
    COLORS = {'standard': Neon.CYAN.norm, 'exchangeable': Neon.GREEN.norm, 'attention': Neon.RED.norm}
    LINESTYLES = {'standard': '--', 'exchangeable': '-.', 'attention': '-'}
    NAMES = {'standard': 'RegretNet', 'exchangeable': 'EquivariantNet', 'attention': 'RegretFormer'}

    TAGS = {'revenue': 'Validation_revenue.csv', 'regret': 'Validation_regret_grad.csv',
            'revenue_train': 'Train_revenue.csv', 'regret_train': 'Train_regret.csv'}

    table_from_csv()

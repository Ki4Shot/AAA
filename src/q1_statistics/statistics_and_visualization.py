import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Microsoft YaHei', font_scale=1.1)

PROCESSED_DIR = os.path.join('data', 'processed')
Q1_FIGURE_DIR = os.path.join('results', 'figures', 'q1')
os.makedirs(Q1_FIGURE_DIR, exist_ok=True)

BUILDINGS = ['地点1', '地点2']

for building in BUILDINGS:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'cleaned_{building}.csv'), parse_dates=['时间'])
    # 1. 室内温度时序图
    plt.figure(figsize=(16,4))
    plt.plot(df['时间'], df['室内平均温度(℃)'], label='室内平均温度', linewidth=2)
    plt.xlabel('时间')
    plt.ylabel('室内平均温度(℃)')
    plt.title(f'{building} 室内温度时序图')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f1 = os.path.join(Q1_FIGURE_DIR, f'{building}_室内温度时序.png')
    plt.savefig(f1, dpi=200)
    plt.close()
    # 2. 室内外温度散点图及相关系数
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='环境温度(℃)', y='室内平均温度(℃)', data=df, alpha=0.5, s=40)
    plt.xlabel('环境温度(℃)')
    plt.ylabel('室内平均温度(℃)')
    plt.title(f'{building} 室内外温度相关性')
    corr = df[['环境温度(℃)', '室内平均温度(℃)']].corr().iloc[0,1]
    plt.annotate(f'相关系数: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='left', va='top')
    plt.legend([],[], frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f2 = os.path.join(Q1_FIGURE_DIR, f'{building}_室内外温度相关性.png')
    plt.savefig(f2, dpi=200)
    plt.close()
    print(f'{building} 室内外温度相关系数: {corr:.3f}')
    # 3. 热泵能耗与温差关系
    df['温差'] = df['供温(℃)'] - df['环境温度(℃)']
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='温差', y='热泵功率(kw)', data=df, alpha=0.5, s=40)
    reg = sns.regplot(x='温差', y='热泵功率(kw)', data=df, scatter=False, color='red', line_kws={'linewidth':2})
    plt.xlabel('供水-环境温度差(℃)')
    plt.ylabel('热泵功率(kw)')
    plt.title(f'{building} 热泵能耗与温差关系')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f3 = os.path.join(Q1_FIGURE_DIR, f'{building}_热泵能耗_温差关系.png')
    plt.savefig(f3, dpi=200)
    plt.close()
    # 4. 主要字段分布直方图
    fields = ['室内平均温度(℃)', '环境温度(℃)', '供温(℃)', '回温(℃)', '热泵功率(kw)']
    for col in fields:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], bins=40, kde=True, color='steelblue', linewidth=1.5)
            plt.xlabel(col)
            plt.title(f'{building} {col} 分布')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            f4 = os.path.join(Q1_FIGURE_DIR, f'{building}_{col}_分布.png')
            plt.savefig(f4, dpi=200)
            plt.close()
            print(f'已保存: {f4}') 
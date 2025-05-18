import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = os.path.join('data', 'processed')
Q4_FIGURE_DIR = os.path.join('results', 'figures', 'q4')
Q4_RESULT_DIR = os.path.join('results', 'q4')
Q4_DATA_DIR = os.path.join(PROCESSED_DIR, 'q4')
os.makedirs(Q4_FIGURE_DIR, exist_ok=True)
os.makedirs(Q4_RESULT_DIR, exist_ok=True)
os.makedirs(Q4_DATA_DIR, exist_ok=True)
PARAMS_PATH = os.path.join(PROCESSED_DIR, 'q2', 'thermal_model_params.csv')
RESULT_PATH = os.path.join(Q4_DATA_DIR, 'control_strategy_results.csv')

BUILDINGS = ['地点1', '地点2']
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))
DAY_PRICE = 1.2
NIGHT_PRICE = 0.6

params_df = pd.read_csv(PARAMS_PATH)
results = []

for building in BUILDINGS:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'cleaned_{building}.csv'), parse_dates=['时间'])
    df = df.sort_values('时间').reset_index(drop=True)
    p = params_df[params_df['建筑'] == building].iloc[0]
    a, b, c, d = p['a'], p['b'], p['c'], p['d']
    # 策略1：恒温控制（目标20℃）
    T_set_const = 20.0
    T_sim_const = [df.loc[0, '室内平均温度(℃)']]
    Q_list_const = []
    for i in range(1, len(df)):
        Tout = df.loc[i-1, '环境温度(℃)']
        T_last = T_sim_const[-1]
        # 反推所需Q使T趋近目标
        Q = (T_set_const - a*T_last - b*Tout - d) / c
        Q = max(Q, 0)  # 不允许负功率
        Q_list_const.append(Q)
        T_new = a*T_last + b*Tout + c*Q + d
        T_sim_const.append(T_new)
    # 策略2：分时控温（夜间21℃，白天19℃）
    T_sim_time = [df.loc[0, '室内平均温度(℃)']]
    Q_list_time = []
    for i in range(1, len(df)):
        hour = df.loc[i-1, '时间'].hour
        T_set = 21.0 if hour in NIGHT_HOURS else 19.0
        Tout = df.loc[i-1, '环境温度(℃)']
        T_last = T_sim_time[-1]
        Q = (T_set - a*T_last - b*Tout - d) / c
        Q = max(Q, 0)
        Q_list_time.append(Q)
        T_new = a*T_last + b*Tout + c*Q + d
        T_sim_time.append(T_new)
    # 计算能耗与电费
    Q_arr_const = np.array(Q_list_const)
    Q_arr_time = np.array(Q_list_time)
    hours = df['时间'][1:]
    price_const = [(NIGHT_PRICE if t.hour in NIGHT_HOURS else DAY_PRICE) for t in hours]
    price_time = price_const
    fee_const = np.sum(Q_arr_const * price_const)
    fee_time = np.sum(Q_arr_time * price_time)
    # 结果保存
    results.append({
        '建筑': building,
        '恒温策略总能耗(kWh)': Q_arr_const.sum(),
        '恒温策略总电费(元)': fee_const,
        '分时控温总能耗(kWh)': Q_arr_time.sum(),
        '分时控温总电费(元)': fee_time
    })
    # 可视化
    plt.figure(figsize=(16,4))
    plt.plot(df['时间'], df['室内平均温度(℃)'], label='历史室温')
    plt.plot(df['时间'], T_sim_const, label='恒温策略仿真')
    plt.plot(df['时间'], T_sim_time, label='分时控温仿真')
    plt.xlabel('时间')
    plt.ylabel('室内温度(℃)')
    plt.title(f'{building} 控温策略温度曲线')
    plt.legend()
    fig1 = os.path.join(Q4_FIGURE_DIR, f'{building}_控温策略_温度曲线.png')
    plt.savefig(fig1, dpi=200)
    plt.close()
    # 能耗对比
    plt.figure(figsize=(8,4))
    plt.plot(hours, Q_arr_const, label='恒温策略功率')
    plt.plot(hours, Q_arr_time, label='分时控温功率')
    plt.xlabel('时间')
    plt.ylabel('热泵功率(kW)')
    plt.title(f'{building} 控温策略能耗对比')
    plt.legend()
    fig2 = os.path.join(Q4_FIGURE_DIR, f'{building}_控温策略_能耗对比.png')
    plt.savefig(fig2, dpi=200)
    plt.close()
    print(f'{building} 策略仿真完成，图表已保存: {fig1}, {fig2}')
# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
print(f'控温策略仿真结果已保存: {RESULT_PATH}') 
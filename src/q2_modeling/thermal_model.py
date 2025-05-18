import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = os.path.join('data', 'processed')
Q2_FIGURE_DIR = os.path.join('results', 'figures', 'q2')
os.makedirs(Q2_FIGURE_DIR, exist_ok=True)
PARAMS_PATH = os.path.join(PROCESSED_DIR, 'q2', 'thermal_model_params.csv')
os.makedirs(os.path.join(PROCESSED_DIR, 'q2'), exist_ok=True)

BUILDINGS = ['地点1', '地点2']
param_records = []

for building in BUILDINGS:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'cleaned_{building}.csv'), parse_dates=['时间'])
    # 构建特征与标签
    df = df.dropna(subset=['室内平均温度(℃)', '环境温度(℃)', '热泵功率(kw)'])
    df = df.sort_values('时间').reset_index(drop=True)
    # 差分建模：T_in(t+1) = a*T_in(t) + b*T_out(t) + c*Q_in(t) + d
    X = np.stack([
        df['室内平均温度(℃)'][:-1].values,
        df['环境温度(℃)'][:-1].values,
        df['热泵功率(kw)'][:-1].values,
        np.ones(len(df)-1)
    ], axis=1)
    y = df['室内平均温度(℃)'][1:].values
    # 拟合
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    # 记录参数
    param_records.append({
        '建筑': building,
        'a': reg.coef_[0],
        'b': reg.coef_[1],
        'c': reg.coef_[2],
        'd': reg.intercept_,
        'RMSE': rmse,
        'R2': r2
    })
    # 可视化
    plt.figure(figsize=(16,4))
    plt.plot(df['时间'][1:], y, label='真实室内温度')
    plt.plot(df['时间'][1:], y_pred, label='模型拟合')
    plt.xlabel('时间')
    plt.ylabel('室内平均温度(℃)')
    plt.title(f'{building} 热力学模型拟合效果')
    plt.legend()
    fig_path = os.path.join(Q2_FIGURE_DIR, f'{building}_热力学模型拟合.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f'{building} 拟合完成，R2={r2:.3f}，RMSE={rmse:.3f}，图表已保存: {fig_path}')
# 保存参数
params_df = pd.DataFrame(param_records)
params_df.to_csv(PARAMS_PATH, index=False, encoding='utf-8-sig')
print(f'参数与性能指标已保存: {PARAMS_PATH}') 
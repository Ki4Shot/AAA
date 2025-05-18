import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join('data', 'processed')
Q3_FIGURE_DIR = os.path.join('results', 'figures', 'q3')
Q3_RESULT_DIR = os.path.join('results', 'q3')
os.makedirs(Q3_FIGURE_DIR, exist_ok=True)
os.makedirs(Q3_RESULT_DIR, exist_ok=True)
PARAMS_PATH = os.path.join(PROCESSED_DIR, 'q2', 'thermal_model_params.csv')

BUILDINGS = ['地点1', '地点2']
FORECAST_H = 4

# 读取热力学模型参数
def get_thermal_params(building):
    params = pd.read_csv(PARAMS_PATH, index_col='建筑')
    row = params.loc[building]
    return row['a'], row['b'], row['c'], row['d']

# 物理模型4小时递推预测
def rc_forecast(df, a, b, c, d, forecast_h=4):
    preds = []
    for i in range(len(df) - forecast_h):
        T = df['室内平均温度(℃)'].iloc[i]
        for h in range(forecast_h):
            T = a * T + b * df['环境温度(℃)'].iloc[i+h] + c * df['热泵功率(kw)'].iloc[i+h] + d
        preds.append(T)
    return np.array(preds)

# XGBoost 4小时滚动预测
def xgb_forecast(df, forecast_h=4):
    features = ['室内平均温度(℃)', '环境温度(℃)', '热泵功率(kw)']
    X, y = [], []
    for i in range(len(df) - forecast_h):
        X.append(df[features].iloc[i].values)
        y.append(df['室内平均温度(℃)'].iloc[i+forecast_h])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, model

for building in BUILDINGS:
    print(f'\n===== {building} 4小时预测 =====')
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'cleaned_{building}.csv'), parse_dates=['时间'])
    df = df.dropna(subset=['室内平均温度(℃)', '环境温度(℃)', '热泵功率(kw)'])
    df = df.sort_values('时间').reset_index(drop=True)
    # 物理模型预测
    a, b, c, d = get_thermal_params(building)
    rc_preds = rc_forecast(df, a, b, c, d, FORECAST_H)
    rc_true = df['室内平均温度(℃)'].iloc[FORECAST_H:].values
    rc_rmse = np.sqrt(mean_squared_error(rc_true, rc_preds))
    rc_r2 = r2_score(rc_true, rc_preds)
    print(f'RC模型: RMSE={rc_rmse:.3f}, R2={rc_r2:.3f}')
    # XGBoost预测
    y_test, y_pred, xgb_model = xgb_forecast(df, FORECAST_H)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    xgb_r2 = r2_score(y_test, y_pred)
    print(f'XGBoost: RMSE={xgb_rmse:.3f}, R2={xgb_r2:.3f}')
    # 保存结果
    result_df = pd.DataFrame({
        '时间': df['时间'].iloc[-len(rc_preds):].values,
        '真实值': rc_true[-len(rc_preds):],
        'RC预测': rc_preds[-len(rc_preds):],
        'XGBoost预测': np.concatenate([np.full(len(rc_true)-len(y_pred), np.nan), y_pred])
    })
    result_path = os.path.join(Q3_RESULT_DIR, f'{building}_4h_forecast.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    # 绘图
    plt.figure(figsize=(16,5))
    plt.plot(result_df['时间'], result_df['真实值'], label='真实值', color='black')
    plt.plot(result_df['时间'], result_df['RC预测'], label='RC模型预测', linestyle='--')
    plt.plot(result_df['时间'], result_df['XGBoost预测'], label='XGBoost预测', linestyle=':')
    plt.xlabel('时间')
    plt.ylabel('室内平均温度(℃)')
    plt.title(f'{building} 4小时温度预测对比')
    plt.legend()
    fig_path = os.path.join(Q3_FIGURE_DIR, f'{building}_4h_forecast_compare.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f'预测结果与对比图已保存: {result_path}, {fig_path}') 
import os
import pandas as pd

PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_DIR = PROCESSED_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUILDINGS = ['地点1', '地点2']

for building in BUILDINGS:
    in_path = os.path.join(PROCESSED_DIR, f'merged_{building}.csv')
    df = pd.read_csv(in_path, parse_dates=['时间'])
    print(f'\n==== {building} 缺失值统计 ===')
    print(df.isnull().sum())
    # 主要字段线性插值填补
    for col in ['供温(℃)', '回温(℃)', '设定温度(℃)', '环境温度(℃)', '热泵功率(kw)', '热量(kw)', '室内平均温度(℃)']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    print(f'\n==== {building} 清洗后样例 ===')
    print(df.head())
    # 保存
    out_path = os.path.join(OUTPUT_DIR, f'cleaned_{building}.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存: {out_path}')
    # 描述性统计
    print(f'\n==== {building} 主要字段描述性统计 ===')
    print(df.describe()) 
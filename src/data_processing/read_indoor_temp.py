import os
import pandas as pd
from glob import glob

DATA_DIR = os.path.join('.', '附件2')
OUTPUT_DIR = os.path.join('data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUILDINGS = ['地点1', '地点2']

for building in BUILDINGS:
    folder = os.path.join(DATA_DIR, building, '室内温度采集数据')
    all_dfs = []
    for file in glob(os.path.join(folder, '*.xlsx')):
        try:
            df = pd.read_excel(file, usecols=['采集时间', '测点温度(℃)'])
            df = df.dropna(subset=['采集时间', '测点温度(℃)'])
            df['采集时间'] = pd.to_datetime(df['采集时间'], errors='coerce')
            df = df.dropna(subset=['采集时间'])
            all_dfs.append(df)
        except Exception as e:
            print(f'文件{file}读取失败: {e}')
    if not all_dfs:
        print(f'{building}无有效数据')
        continue
    all_data = pd.concat(all_dfs, ignore_index=True)
    # 按小时聚合均值
    all_data['hour'] = all_data['采集时间'].dt.floor('H')
    mean_temp = all_data.groupby('hour')['测点温度(℃)'].mean().reset_index()
    mean_temp = mean_temp.rename(columns={'hour': '时间', '测点温度(℃)': '室内平均温度(℃)'})
    print(f'\n==== {building} 室内温度均值样例 ===')
    print(mean_temp.head())
    # 保存
    out_path = os.path.join(OUTPUT_DIR, f'indoor_temp_mean_{building}.csv')
    mean_temp.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存: {out_path}') 
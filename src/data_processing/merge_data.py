import os
import pandas as pd
from glob import glob

DATA_DIR = os.path.join('.', '附件2')
PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_DIR = PROCESSED_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUILDINGS = ['地点1', '地点2']

for building in BUILDINGS:
    # 1. 合并所有年份的供热历史数据
    his_folder = os.path.join(DATA_DIR, building, '供热历史数据')
    his_files = glob(os.path.join(his_folder, '*.xlsx'))
    his_dfs = []
    for file in his_files:
        try:
            df = pd.read_excel(file)
            df = df[['时间', '供温(℃)', '回温(℃)', '设定温度(℃)', '环境温度(℃)', '热泵功率(kw)', '热量(kw)']]
            df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
            df = df.dropna(subset=['时间'])
            his_dfs.append(df)
        except Exception as e:
            print(f'文件{file}读取失败: {e}')
    if not his_dfs:
        print(f'{building}无有效供热历史数据')
        continue
    his_data = pd.concat(his_dfs, ignore_index=True)
    his_data = his_data.sort_values('时间')
    # 2. 读取室内温度均值
    indoor_path = os.path.join(PROCESSED_DIR, f'indoor_temp_mean_{building}.csv')
    indoor = pd.read_csv(indoor_path)
    indoor['时间'] = pd.to_datetime(indoor['时间'], errors='coerce')
    # 3. 按时间（小时）对齐合并
    merged = pd.merge(his_data, indoor, on='时间', how='outer')
    merged = merged.sort_values('时间').reset_index(drop=True)
    # 4. 输出样例并保存
    print(f'\n==== {building} 融合数据样例 ===')
    print(merged.head())
    out_path = os.path.join(OUTPUT_DIR, f'merged_{building}.csv')
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存: {out_path}') 
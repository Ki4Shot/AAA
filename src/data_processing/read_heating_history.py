import os
import pandas as pd

DATA_DIR = os.path.join('.', '附件2')

BUILDINGS = ['地点1', '地点2']

for building in BUILDINGS:
    folder = os.path.join(DATA_DIR, building, '供热历史数据')
    print(f'\n==== {building} 供热历史数据 ===')
    for file in os.listdir(folder):
        if file.endswith('.xlsx'):
            path = os.path.join(folder, file)
            print(f'文件: {file}')
            df = pd.read_excel(path, nrows=5)
            print('字段:', list(df.columns))
            print(df.head(), '\n') 
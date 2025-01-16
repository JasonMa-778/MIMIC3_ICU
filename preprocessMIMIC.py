import pandas as pd
import numpy as np
import json
from collections import ChainMap
from os.path import join
import sys
from tqdm import tqdm

# 工具函数：将纳秒转为天
dayconverter = lambda x: np.timedelta64(x, 'ns') / np.timedelta64(1, 'D')

if __name__ == "__main__":
    path = sys.argv[1]  # 数据路径

    # 1. 读取 chartevents 文件
    print('[1/7] Reading chartevents file...')
    try:
        df = pd.read_csv(join(path, 'CHARTEVENTS.csv.gz'), compression='gzip',
                         usecols=['icustay_id', 'itemid', 'valuenum', 'charttime'])
    except FileNotFoundError:
        print(f"❌ 文件 {join(path, 'chartevents.csv.gz')} 不存在，请检查路径。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取 chartevents.csv.gz 出错: {e}")
        sys.exit(1)

    df.dropna(inplace=True)

    # 2. 获取每个 icustay_id 的最早和最晚 charttime
    print('[2/7] Computing admission and discharge times...')
    adm_dates = df.groupby('icustay_id')['charttime'].min()
    exit_dates = df.groupby('icustay_id')['charttime'].max()

    # 3. 计算 icu 停留时间
    print('[3/7] Calculating ICU stay duration...')
    icudates = pd.DataFrame({'intime': adm_dates, 'outtime': exit_dates})
    icudates['duration'] = pd.to_datetime(icudates['outtime']) - pd.to_datetime(icudates['intime'])
    icudates = icudates[
        (icudates['duration'] >= pd.Timedelta(days=1)) &
        (icudates['duration'] <= pd.Timedelta(days=2))
    ]

    # 4. 过滤 ICU 停留时间不在 [1, 2] 天的数据
    print('[4/7] Filtering ICU stays not in [1, 2] days...')
    df = df[df['icustay_id'].isin(icudates.index)]
    df['dt'] = pd.to_datetime(df['charttime']) - pd.to_datetime(df['icustay_id'].map(icudates['intime']))

    # 5. 过滤前 3 小时内的事件
    print('[5/7] Filtering events outside the first 3 hours...')
    df = df[df['dt'] < pd.Timedelta(hours=3)]
    df['dt'] = df['dt'].dt.total_seconds() / 60  # 转换为分钟

    # 转换为字典
    pdf = df.groupby(["icustay_id", "dt"]).apply(lambda x: list(zip(x.itemid, x.valuenum))).reset_index(name='val')
    pdf = pdf.groupby("icustay_id").apply(lambda x: dict(zip(x.dt, x.val))).to_dict()

#    # 6. 过滤事件总数 >100 的记录
#    print('[6/7] Filtering records with >100 events...')
#    res = {}
#    for patient, data in tqdm(pdf.items(), desc="Filtering events"):
#        sorted_keys = sorted(data.keys())
#        limited_data = {}
#        count = 0
#
#        for k in sorted_keys:
#            if count >= 100:
#                break
#            items = data[k][:max(0, 100 - count)]
#            limited_data[k] = items
#            count += len(items)
#
#        res[int(patient)] = limited_data
    # 6. 保留所有事件数据，不做数量限制
    print('[6/7] Keeping all records without event limits...')
    res = {}
    for patient, data in tqdm(pdf.items(), desc="Processing events"):
        res[int(patient)] = data


    # 7. 读取 admissions 和 icustays 文件
    print('[7/7] Reading admissions and icustays...')
    try:
        adm = pd.read_csv(join(path, 'ADMISSIONS.csv.gz'), compression='gzip', usecols=['hadm_id', 'deathtime'])
        icustays = pd.read_csv(join(path, 'ICUSTAYS.csv.gz'), compression='gzip', usecols=['hadm_id', 'icustay_id'])
    except Exception as e:
        print(f"❌ 读取 admissions/icustays 文件失败: {e}")
        sys.exit(1)

    # 建立映射关系
    icustay2hadm = dict(zip(icustays.icustay_id, icustays.hadm_id))
    adm['deathtime'] = adm.deathtime.isna().astype(int)
    hadm2label = dict(zip(adm.hadm_id, adm.deathtime))

    icustay2label = {icu_k: hadm2label.get(hadm_k, 0) for icu_k, hadm_k in icustay2hadm.items()}
    restricted_icustay2label = {k: v for k, v in icustay2label.items() if k in res}

    # 保存 JSON 文件
    print('Saving JSON files...')
    with open('events.json', "w") as f:
        json.dump(res, f, sort_keys=True)

    with open('targets.json', "w") as f:
        json.dump(restricted_icustay2label, f, sort_keys=True)

    print("✅ Preprocessing complete!")

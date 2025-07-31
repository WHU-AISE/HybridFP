
import numpy as np
from tqdm import tqdm
import pickle as pkl
import polars as pl
import stumpy
from common import *
from stumpy import config
from robustperiod import robust_period_full

config.STUMPY_EXCL_ZONE_DENOM = 1

MID_DATA = "../mid_data/AzureData"
RESULT_DATA = "../result/AzureData"

with open(MID_DATA + "/arrcount.pkl", "rb") as file:
    train_func_arrcount, test_func_arrcount = pkl.load(file)
with open(MID_DATA + '/variables_checkpoint.pkl', "rb") as file:
    train_func_ids, test_func_arrcount, func_class, func_lst, func_corr_lst, test_func_corr, test_func_corr_perform = pkl.load(
        file)

# 过滤得可预测函数
df = pl.read_csv(MID_DATA + "/func_info.csv")
pe_df = df.filter(pl.col('PE') > 0.7)    

df_union = pl.concat([pe_df]).unique()
all_predictable_func = df_union.select('Function').to_numpy().flatten().tolist()
all_predictable_func = list(map(lambda x: str(x), all_predictable_func))
print(len(all_predictable_func))

other_func = []
predictable_func = []
regular_func = []
for func in test_func_arrcount:
    if func not in all_predictable_func:
        other_func.append(func)  
    else:
        predictable_func.append(func)

print(len(test_func_arrcount))
print(len(other_func), len(predictable_func))

other_func_motif = {}
for func in tqdm(other_func):
    if func not in train_func_arrcount:
        continue    
    train_data = np.array(train_func_arrcount[func][-1440*3:]).astype(float)
    m = 1440
    time1 = time.time()
    mp = stumpy.stump(train_data, m)
    time2 = time.time()
    print(f"Time taken: {time2 - time1} seconds")
    motif_idx = np.argsort(mp[:,0])[0]
    # print(f"The motif is located at index {motif_idx}, with a value of {mp[motif_idx,]}")
    nearest_neighbor_idx = mp[motif_idx, 1] # Find the index of the nearest neighbor
    other_func_motif[func] = (motif_idx, nearest_neighbor_idx, mp[motif_idx, 0], m)
    # print(f"The nearest neighbor is located at index {nearest_neighbor_idx}, with a  value of {mp[nearest_neighbor_idx,]}")

with open(MID_DATA + "/other_func_motif.pkl", "wb") as file:
    pkl.dump(other_func_motif, file)

other_func_period = {}
# 周期性挖掘
for func in tqdm(other_func):
    if func_class[func].type == REGULAR:
        other_func_period[func] = [func_class[func].pred_interval[0]+1]
    else:
        data = train_func_arrcount[func]
        data_recent = np.array(data[-1440*3:])
        is_all_zero = np.all(np.array(data_recent) == 0)
        if is_all_zero:
            other_func_period[func] = []
            continue
        lmb = 1e+6
        lmb = 1e+6
        c = 2
        num_wavelets = 8
        zeta = 1.345
        periods, W, bivar, periodograms, p_vals, ACF = robust_period_full(
        data_recent, 'db10', num_wavelets, lmb, c, zeta)
        other_func_period[func] = periods

# 保存周期性挖掘结果
with open(MID_DATA +'/other_func_periods.pkl', 'wb') as file:
    pkl.dump(other_func_period, file)

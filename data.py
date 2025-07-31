import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import pickle
from plot import *
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import adfuller
import scipy.signal as signal
from mne.time_frequency import psd_array_multitaper
from statsmodels.tsa.stattools import acf
from math import factorial
import time

DATASET_DIR = "./azure-data/"
PERCENTILE_NAME = ["P5", "P25", "P50", "P75", "P99"]
COLOR_NAME = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred']


def save_df(save_dir: str, df:pl.DataFrame, file_name: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df.write_csv(os.path.join(save_dir, file_name))

# 将14天负载数据按照函数ID进行聚合
def load_func_wo_timer(dataset_dir: str,
                 num_days: int) -> pl.DataFrame:

    for day in tqdm(range(1, num_days + 1)):
        invocations_per_function = (dataset_dir + 'invocations_per_function_md.anon.d0' + str(day) + '.csv') \
            if day < 10 else (dataset_dir + 'invocations_per_function_md.anon.d' + str(day) + '.csv')
        df_t = pl.read_csv(invocations_per_function)
        df_t = df_t.filter(pl.col("Trigger") != "timer")  #过滤去除定时器触发的函数
        df_t = df_t.select(['HashFunction'] + df_t.columns[4:])
        funcIDs = df_t['HashFunction'].to_list()
        df_t = df_t.transpose(include_header=True,column_names=funcIDs)
        df_t = df_t.select(funcIDs)[1:]

        joined_df = pl.concat([joined_df, df_t], how="diagonal") if day > 1 else df_t

    joined_df.columns = [str(i) for i in range(len(joined_df.columns))]
    start_t = datetime(2024, 1, 1)
    end_t = start_t + timedelta(days=num_days, minutes=-1)
    time = pl.datetime_range(start=start_t, end=end_t, interval="1m",time_unit='ms',eager=True).alias("time")
    joined_df = joined_df.with_columns(pl.col(joined_df.columns).cast(pl.Int32))
    joined_df = joined_df.insert_column(index=0, column=time)
    return joined_df


# 将14天负载数据按照函数ID进行聚合
def load_all_func(dataset_dir: str,
                 num_days: int) -> pl.DataFrame:

    for day in tqdm(range(1, num_days + 1)):
        invocations_per_function = (dataset_dir + 'invocations_per_function_md.anon.d0' + str(day) + '.csv') \
            if day < 10 else (dataset_dir + 'invocations_per_function_md.anon.d' + str(day) + '.csv')
        df_t = pl.read_csv(invocations_per_function)
        df_t = df_t.select(['HashFunction'] + df_t.columns[4:])
        funcIDs = df_t['HashFunction'].to_list()
        df_t = df_t.transpose(include_header=True,column_names=funcIDs)
        df_t = df_t.select(funcIDs)[1:]

        joined_df = pl.concat([joined_df, df_t], how="diagonal") if day > 1 else df_t

    # joined_df.columns = [str(i) for i in range(len(joined_df.columns))]
    start_t = datetime(2024, 1, 1)
    end_t = start_t + timedelta(days=num_days, minutes=-1)
    time = pl.datetime_range(start=start_t, end=end_t, interval="1m",time_unit='ms',eager=True).alias("time")
    joined_df = joined_df.with_columns(pl.col(joined_df.columns).cast(pl.Int32))
    joined_df = joined_df.insert_column(index=0, column=time)
    joined_df = joined_df.fill_nan(0)
    return joined_df


# 筛选清洗函数负载数据：
def filter(dataset_dir:str, csv_name:str, describe_csv_name:str):
    df_target = pl.read_csv(os.path.join(dataset_dir,csv_name))
    # df_target = df_target.fill_null(0)  #补0
    df_describe = pd.read_csv(os.path.join(dataset_dir, describe_csv_name))
    
    columns_to_select = df_describe.iloc[0,2:] > 2880   # 1.筛选至少包含三天负载数据的函数
    selected_columns = columns_to_select[columns_to_select].index.tolist()
    df_target = df_target.select(["time"] + selected_columns)

    return df_target

# 获取每列的峰度值（衡量数据分布的尾部厚度和峰顶的尖锐程度相对于正态分布的情况。）
def get_kurtosis(df:pl.DataFrame) -> list:
    kurtosis_values = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        kurtosis_value = kurtosis(data_c, fisher=False)
        kurtosis_values.append((c, kurtosis_value))
    return kurtosis_values

# 获取每列的标准差
def get_std(df:pl.DataFrame) -> list:
    std_values = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        std_values.append((c, data_c.std()))
        
    return std_values    

# 获取每列的峰值比率
def get_peak_ratio(df:pl.DataFrame) -> list:
    peak_ratio = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        mean = data_c.mean()
        max = data_c.max()
        peak_ratio.append((c, max/mean) if mean != 0 else (c, 0))
    return peak_ratio

# 获取每列异常值比率
def get_outlier_proportion(df:pl.DataFrame) -> list:
    outlier_proportion = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        mean = data_c.mean()
        std = data_c.std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        data_outliers = data_c[(data_c < lower_bound) | (data_c > upper_bound)]
        proportion = len(data_outliers) / len(data_c)
        outlier_proportion.append((c, proportion))
    return outlier_proportion

# 获取每列的变异系数CV
def get_CV(df:pl.DataFrame) -> list:
    CVs = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]   
        mean = data_c.mean()
        std = data_c.std()
        cv = std/mean
        CVs.append((c, cv))
    return CVs

# 获取每列的相邻两次调用的IAT
def get_CV_IAT(df:pl.DataFrame) -> list:
    CV_IATs = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        
        nonzero_idx = np.where(data_c != 0)[0]
        
        if len(nonzero_idx) < 2:
            CV_IATs.append((c, np.nan))
        else:
            iats = (nonzero_idx[1:] - nonzero_idx[:-1]).tolist()
            cv_iat = np.std(iats) / np.mean(iats)
            CV_IATs.append((c, cv_iat))
    return CV_IATs


# 平稳性 （ADF单位根检验）：要求时间序列的均值和方差保持不变，并且任意两个时期的协方差只依赖于这两个时期之间的时间差
def get_adfuller(df:pl.DataFrame) -> list:
    result = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        if data_c.max() == data_c.min():
            result.append((c, -1))
            continue
        adfuller_result = adfuller(data_c, autolag= "AIC")
        is_stationary = 1 if adfuller_result[1] < 0.05 else 0
        result.append((c, is_stationary))
    return result

# 周期性 PSD频率
def get_periodicity(df:pl.DataFrame) -> list:
    result = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        fs = 1 / 60  #数据采样频率
        # frequencies, psd_values = signal.welch(data_c, fs, nperseg=1440 * 7)
        frequencies, psd_values = signal.welch(data_c, fs, nperseg=1440 * 7)
        max_psd_index = np.argmax(psd_values)
        peak_frequency = frequencies[max_psd_index]     #the frequency at the peak of PSD
        peak_psd = psd_values[max_psd_index]

        peak_psd_normalized = peak_psd / np.sum(psd_values)
        result.append((c, (1/peak_frequency)/60, peak_psd_normalized))

    return result

# 复杂性 排列熵
def get_entropy(df:pl.DataFrame):
    result = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]
        time1 = time.time()
        pe = permutation_entropy(data_c)
        time2 = time.time()
        print(f"Time taken: {time2 - time1} seconds")
        result.append((c, pe, time2 - time1))
    return result

def _embed(x, order=3, delay=1):
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

# 排列熵
def permutation_entropy(x, order=3, delay=1, normalize=True):
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')#【1】
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)#【2】
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()#【3】
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def get_reqs_per_day(df:pl.DataFrame) -> pl.DataFrame:
    result = []
    for c in df.columns:
        data_c = df[c].to_numpy().astype(float)
        data_c = data_c[~np.isnan(data_c)]   
        grouped_sums = np.sum(data_c.reshape(-1, 1440), axis=1)
        mean_reqs = int(np.mean(grouped_sums))
        result.append((c, mean_reqs))
    df_result = pl.DataFrame(result,schema=['Function','Reqs_Per_Day'],orient="row")
    return df_result

if __name__ == "__main__":    
    pass
import os
import sys
import pandas as pd
import numpy as np
from math import ceil
import pickle

sys.path.insert(1, '/home/sigao/xyf/azurefunctions-dataset2019/')

save_dir = "/home/sigao/xyf/azurefunctions-dataset2019/"
store = "/home/sigao/xyf/azurefunctions-dataset2019/"

datapath = "/home/sigao/xyf/azurefunctions-dataset2019/"
durations = "function_durations_percentiles.anon.d13.csv"
invocations = "invocations_per_function_md.anon.d13.csv"
mem_fnames = "app_memory_percentiles.anon.d13.csv"
func_info_file = "func_info.csv" # 新增函数信息文件

quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_data(file_path, dtype_dict):
    df = pd.read_csv(file_path, dtype=dtype_dict)
    df.index = df["HashFunction"]
    df = df.drop_duplicates("HashFunction")
    return df


def compute_interarrival_and_cv(invocation_counts):
    interarrival_times = []
    zero_counter = 0
    for value in invocation_counts:
        if value == 0:
            zero_counter += 1
        else:
            # 当前时间单位有调用，第一次调用在开始时刻
            if zero_counter > 0:
                # 之前有空闲期：第一次调用间隔 = 零值数量
                interarrival_times.append(zero_counter)
                # 当前时间单位内的其他调用
                if value > 1:
                    interval = 1.0 / value  # 在当前时间单位内均匀分布
                    for i in range(1, value):
                        interarrival_times.append(interval)
            else:
                # 连续调用：第一次调用间隔 = 0（紧接上一个调用）
                if value == 1:
                    interarrival_times.append(0)
                else:
                    interarrival_times.append(0)  # 第一次调用间隔为0
                    interval = 1.0 / value  # 在当前时间单位内均匀分布
                    for i in range(1, value):
                        interarrival_times.append(interval)
            zero_counter = 0
    # 计算统计量
    if len(interarrival_times) == 0:
        return 0, 0
    mean_interarrival_time = np.mean(interarrival_times)
    std_interarrival_time = np.std(interarrival_times)
    cv_interarrival_time = std_interarrival_time / mean_interarrival_time if mean_interarrival_time > 0 else 0
    return mean_interarrival_time, cv_interarrival_time


def gen_traces1():
    global durations
    global invocations
    global memory

    def divive_by_func_num(row):
        return ceil(row["AverageAllocatedMb"] / group_by_app[row["HashApp"]])

    col_types = {str(i): int for i in range(1, 481)} # 修改这里，以适应480分钟数据
    durations_file = os.path.join(datapath, durations)
    invocations_file = os.path.join(datapath, invocations)
    func_info_path = os.path.join(datapath, func_info_file)

    durations = load_data(durations_file, col_types)
    invocations = load_data(invocations_file, col_types)

    # 加载 func_info.csv
    try:
        func_info = pd.read_csv(func_info_path)
        func_info.index = func_info["Function"]
        func_info = func_info.drop_duplicates("Function")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {func_info_path}")
        return
    except Exception as e:
        print(f"加载 func_info.csv 时出错: {e}")
        return

    # 将 func_info 中的 PE 值合并到 invocations DataFrame
    invocations = invocations.merge(func_info[['PE']], left_index=True, right_index=True, how='left')
    invocations = invocations.dropna(subset=['PE']) # 移除没有PE值的函数

    group_by_app = durations.groupby("HashApp").size()

    sums = invocations.loc[:, "1":"480"].sum(axis=1)
    #print(sums)

    invocations = invocations[sums > 1]  # action must be invoked at least twice
    invocations = invocations.drop_duplicates("HashFunction")

    # Create a DataFrame to store average interarrival time and CV for each function
    stats = pd.DataFrame(columns=['HashFunction', 'AverageInterarrivalTime', 'CV', 'PE'])

    # Iterate over each function and compute the average interarrival time and CV
    for index, row in invocations.iterrows():
        invocation_counts = row["1":"480"].values # 修改这里，匹配1到480分钟数据
        #print(invocation_counts)
        average_interarrival_time, cv_interarrival_time = compute_interarrival_and_cv(invocation_counts)
        stats = pd.concat([stats, pd.DataFrame([{'HashFunction': index,
                                                 'AverageInterarrivalTime': average_interarrival_time,
                                                 'CV': cv_interarrival_time,
                                                 'PE': row['PE'] # 添加PE值
                                                 }])], ignore_index=True)

    # Filter the DataFrame to find functions that meet the specified conditions
    suitable_functions = stats[(stats['CV'] <= 4) & (stats['CV'] >= 1) & (stats['AverageInterarrivalTime'] >= 0.01) & (
                stats['AverageInterarrivalTime'] <=10)]

    if suitable_functions.empty:
        print('No suitable function found after initial filtering.')
        return

    # 根据PE值进行分位数筛选
    selected_functions = pd.DataFrame()
    pe_quantiles = stats['PE'].quantile(quantiles)

    for i in range(len(quantiles) - 1):
        lower_bound = pe_quantiles.iloc[i]
        upper_bound = pe_quantiles.iloc[i+1]

        # 最后一个区间包含上限，其他区间不包含上限以避免重复
        if i == len(quantiles) - 2:
            quantile_group = suitable_functions[
                (suitable_functions['PE'] >= lower_bound) & (suitable_functions['PE'] <= upper_bound)
            ]
        else:
            quantile_group = suitable_functions[
                (suitable_functions['PE'] >= lower_bound) & (suitable_functions['PE'] < upper_bound)
            ]

        if not quantile_group.empty:
            # 从每个区间中随机选择4个函数
            num_to_sample = min(4, len(quantile_group))
            selected_functions = pd.concat([selected_functions, quantile_group.sample(n=num_to_sample)])
        else:
            print(f"Warning: No suitable functions found in PE quantile range [{lower_bound:.2f}, {upper_bound:.2f}).")

    if selected_functions.empty:
        print('No functions selected after PE quantile filtering.')
        return

    # 如果选中的函数少于16个，尝试从所有 suitable_functions 中随机补充
    if len(selected_functions) < 16 and not suitable_functions.empty:
        remaining_needed = 16 - len(selected_functions)
        # 排除已经选中的函数，从剩余的 suitable_functions 中随机选择
        remaining_functions = suitable_functions.drop(selected_functions.index, errors='ignore')
        if not remaining_functions.empty:
            num_to_sample = min(remaining_needed, len(remaining_functions))
            selected_functions = pd.concat([selected_functions, remaining_functions.sample(n=num_to_sample)])
    
    # 确保最终选择的函数数量不超过16个
    if len(selected_functions) > 16:
        selected_functions = selected_functions.sample(n=16)

    print(f'Final number of selected functions: {len(selected_functions)}')
    print(f'Selected functions: {selected_functions}')
    print(selected_functions['AverageInterarrivalTime'])

    # 确保只使用实际选择的函数来构建 selected_function_invocations
    selected_function_invocations = {}
    for idx, row in selected_functions.iterrows():
        func_hash = row["HashFunction"]
        if func_hash in invocations.index:
            selected_function_invocations[func_hash] = invocations.loc[func_hash, "1":"480"]
        else:
            print(f"Warning: Invocation data not found for selected function {func_hash}")


    with open('selected_functions_240min.pkl', 'wb') as f:
        pickle.dump(selected_functions, f)

    with open('selected_function_invocations_240min.pkl', 'wb') as f:
        pickle.dump(selected_function_invocations, f)

    print("数据已更新并保存到 selected_functions_480min.pkl 和 selected_function_invocations_480min.pkl")

gen_traces1()

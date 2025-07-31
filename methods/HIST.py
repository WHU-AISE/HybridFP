import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle as pkl
import pmdarima as pm
from common import *

# HIST参数设置
RANGE = 4 * 60
HEAD = 0.05
TAIL = 0.99

OOB_THRESHOLD = 0.8
HIST_CV_THRESHOLD = 2
HISTOGRAM_MARGIN = 0.1
ARIMA_MARGIN = 0.15


def get_ITs(arr):
    # 保存非零元素的索引
    non_zero_indices = [i for i, value in enumerate(arr) if value != 0]

    # 计算相邻非零元素之间的间隔
    ITs = []
    for i in range(1, len(non_zero_indices)):
        ITs.append(non_zero_indices[i] - non_zero_indices[i - 1] - 1)

    return ITs


def judge_many_OOB(ITs):
    wb_ITs = [IT for IT in ITs if IT < RANGE]
    too_many_OOB = ((len(ITs) - len(wb_ITs)) / len(ITs)) > OOB_THRESHOLD if len(ITs) >= 3 else False
    return too_many_OOB, wb_ITs


def is_hist_representative(hist):
    # 计算CV
    mean = np.mean(hist)
    std = np.std(hist)
    CV = std / mean

    return CV >= HIST_CV_THRESHOLD


def range_histogram_policy(hist):
    IT_length = sum(hist)
    total = 0
    p_pre_warm = int(IT_length * HEAD)  # 向下取整
    p_keep_alive = int(IT_length * TAIL)  # 向下取整
    flag_pre_warm, flag_keep_alive = False, False

    pre_warm, keep_alive = 0, 0
    for i in range(len(hist)):
        if (hist[i] > 0):
            bin_count = hist[i]
            d_head, d_tail = p_pre_warm - total, p_keep_alive - total
            # Pre-warming window logic
            if not flag_pre_warm and d_head > 0 and d_head <= bin_count:
                flag_pre_warm = True
                if d_head < 0.5 * bin_count:
                    pre_warm = max(0, i - 1)  # Round down (head)
                else:
                    pre_warm = i  # No rounding

            # Keep-alive window logic
            if not flag_keep_alive and d_tail > 0 and d_tail <= bin_count:
                flag_keep_alive = True
                if d_tail < 0.5 * bin_count:
                    keep_alive = i  # Round up (tail)
                else:
                    keep_alive = min(len(hist), i + 1)  # Ensure it doesn't exceed the OOB range

            if flag_keep_alive and flag_pre_warm:
                break

            total += bin_count
    return int(pre_warm * (1 - HISTOGRAM_MARGIN)), min(RANGE, int(keep_alive * (1 + HISTOGRAM_MARGIN)))
    # return pre_warm, keep_alive


if __name__ == "__main__":
    MID_DATA = "../mid_data/AzureData"
    RESULT_DATA = "../result/AzureData"

    # 读取测试集数据并加载函数必要信息
    with open(MID_DATA + "/arrcount.pkl", "rb") as file:
        train_func_arrcount, test_func_arrcount = pkl.load(file)

    ###
    func_class = {}
    with open(MID_DATA + "/train_info_assigned.txt") as rf:    # 所有函数的负载数据 hashID  forget  loadarray
        for line in rf:
            func, type, forget = line.strip().split('\t')
            func_class[func] = func_state(_type=int(type), forget=int(forget))
    print(len(func_class))
    ###

    # 加载测试集函数
    func_lst, func_corr_lst = set(), set()
    num_unseen_func = 0
    for func in func_class:
        if func_class[func].type == CORR:
            func_corr_lst.add(func)
        else:
            func_lst.add(func)

    for func in test_func_arrcount:
        if func in func_class:
            continue
        num_unseen_func += 1
        func_lst.add(func)  # Unseen 函数 训练集中未出现的函数
        func_class[func] = func_state()

    func_lst, func_corr_lst = list(func_lst), list(func_corr_lst)
    print(len(func_class), len(func_lst) + len(func_corr_lst), len(test_func_arrcount), num_unseen_func)

    memory = set()
    test_func = set(test_func_arrcount.keys())

    # 结果记录
    func_cold = {func: 0 for func in test_func}  # 冷启动次数
    func_invok = {func: 0 for func in test_func}  # 调用次数
    func_invok_seq = {func: [] for func in test_func}

    func_invok_bool = {func: 0 for func in test_func}  # 存在调用的单位时间
    func_waste = {func: 0 for func in test_func}  # 内存浪费的单位时间
    print(len(func_cold))
    waste_mem_time = 0

    arima_models = {}
    train_test_funcs = set(train_func_arrcount.keys()) & test_func
    # 记录ITs，构建Histogram，并定义预热和保持窗口
    c=0
    with tqdm(total=len(train_test_funcs)) as pbar:
        for func in train_test_funcs:
            if c % shown_func_num == shown_func_num - 1:
                pbar.update(shown_func_num)
            c += 1
            train_arr = train_func_arrcount[func]
            invok = conj_seq_lst(train_arr, count_invoke=True)
            if(len(invok)==0):
                func_class[func].last_call = 0 - len(train_arr)
                func_class[func].nearest_invoke = 0
                func_class[func].invok_info["mean"] = 0
                func_class[func].invok_info["std"] = 0
                func_class[func].invok_info["invok_nums"] = 0
            else:
                func_class[func].nearest_invoke = train_arr[np.where(np.array(train_arr)>0)[0][-1]]
                func_class[func].last_call = np.where(np.array(train_arr)>0)[0][-1] - len(train_arr)            
                invok_mean = np.mean(np.array(train_arr)[np.where(np.array(train_arr)>0)][0])
                invok_std = np.std(np.array(train_arr)[np.where(np.array(train_arr)>0)][0])
                func_class[func].invok_info["mean"] = invok_mean
                func_class[func].invok_info["std"] = invok_std
                func_class[func].invok_info["invok_nums"] = len(np.where(np.array(train_arr)>0)[0])
    
            ITs = get_ITs(train_arr)
            func_class[func].ITs = ITs
            too_many_OOB, wb_ITs = judge_many_OOB(ITs)  
            if too_many_OOB:    #ITs数据足够且OOB
                if np.var(ITs) < 1e-6:  #ITs 的数据全为相同值,会导致 ARIMA 模型失效
                    next_it_prediction = ITs[0]
                    arima_models[func] = None
                else:
                    # 论文复现使用auto_arima
                    try:
                        model = pm.auto_arima(ITs, seasonal=True, stepwise=True, suppress_warnings=True)
                        next_it_prediction = model.predict(n_periods=1)[0]
                        arima_models[func] = model
                    except Exception as e:
                        print(ITs)
                        print(f"Error during model fitting: {e}")
                        next_it_prediction = np.mean(ITs)              
                pre_warm = next_it_prediction * (1 - ARIMA_MARGIN)
                keep_alive = next_it_prediction * ARIMA_MARGIN * 2
                func_class[func].window_param = (pre_warm, keep_alive)
            else:
                # construct histogram
                for IT in wb_ITs:
                    func_class[func].IT_histogram[IT] += 1
                # judge whether the historgram is representative
                if len(ITs) <= 3 or not is_hist_representative(func_class[func].IT_histogram):
                    # Standard keep-alive when the pattern is uncertain
                    func_class[func].window_param = (0, 240)
                else:
                    # Range-limited histogram
                    pre_warm, keep_alive = range_histogram_policy(func_class[func].IT_histogram)
                    func_class[func].window_param = (pre_warm, keep_alive)

    with open(MID_DATA + '/hist_func.pkl','wb') as f:
        pkl.dump((func_class, arima_models), f)

    # Pre_warm for testing at time 0
    for func in train_test_funcs:
        if func_class[func].window_param is not None:
            pre_warm, keep_alive = func_class[func].window_param
            if pre_warm == 0:
                if func_class[func].last_call + keep_alive >= 0:
                    # 上次调用后持续保活至此
                    last_invoke = train_func_arrcount[func][np.where(np.array(train_func_arrcount[func]) > 0)[0][-1]]
                    memory.add(func)
                    func_class[func].load(0, last_invoke)
                    func_class[func].left_keep_alive = func_class[func].last_call + keep_alive + 1
            else:
                distance = (0 - func_class[func].last_call) - 1
                if (distance % (pre_warm + keep_alive)) - pre_warm <= 0:
                    # 此时处于Pre-warm窗口
                    func_class[func].left_pre_warm = pre_warm - (distance % (pre_warm + keep_alive))
                else:
                    # 此时处于Keep-alive窗口
                    memory.add(func)
                    # func_class[func].load(0, int(func_class[func].invok_info["mean"]))
                    func_class[func].load(0, func_class[func].nearest_invoke)
                    func_class[func].left_keep_alive = pre_warm + keep_alive - (distance % (pre_warm + keep_alive))
        
        
    # 模拟
    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            if i % 10 == 9:
                pbar.update(10)
            # random.shuffle(func_corr_lst)
            for func in test_func:  # In case of some functions staying in the memory forever
                pre_warm, keep_alive = func_class[func].window_param if func_class[func].window_param is not None else (0, 240)
                # 根据left_pre_warm和left_keep_alive判断容器状态
                keep_alive_flag = False  # 区别于冷启动的容器
                if func_class[func].left_pre_warm > 0:
                    if func_class[func].state:
                        memory.remove(func)
                        func_class[func].unload()
                else:
                    # left_pre_warm = 0, 预热容器或容器处于保活
                    if not func_class[func].state:
                        memory.add(func)
                        # prewarm_num = func_class[func].invok_info.get("mean", 1)
                        prewarm_num = func_class[func].nearest_invoke if func_class[func].nearest_invoke is not None else 1
                        func_class[func].load(i, int(prewarm_num))
                    else:
                        if func_class[func].left_keep_alive > 0:
                            keep_alive_flag = True
                        else:
                            memory.remove(func)
                            func_class[func].unload()

                if func in test_func_arrcount and test_func_arrcount[func][i] > 0:
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_seq[func].append(test_func_arrcount[func][i])
                    func_invok_bool[func] += 1

                    if not func_class[func].state:  # cold start
                        func_cold[func] += test_func_arrcount[func][i]
                        func_class[func].load(i, test_func_arrcount[func][i])
                        memory.add(func)
                    elif func_class[func].state and func_class[func].containers_num < test_func_arrcount[func][i]:
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num
                        func_class[func].load(i, test_func_arrcount[func][i] - func_class[func].containers_num)
                    elif func_class[func].state and func_class[func].containers_num >= test_func_arrcount[func][i]:  # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]
                        func_waste[func] += func_class[func].containers_num - test_func_arrcount[func][i]

                    # 更新 HIST
                    IT = i - func_class[func].last_call - 1 if func_class[func].last_call is not None else -1
                    func_class[func].ITs.append(IT)
                    if IT == -1:  # 无历史记录的函数
                        None
                    elif IT < RANGE:
                        func_class[func].IT_histogram[IT] += 1
                        if sum(func_class[func].IT_histogram) <= 3 or not is_hist_representative(func_class[func].IT_histogram):
                            # Standard keep-alive when the pattern is uncertain
                            func_class[func].window_param = (0, 240)
                        else:
                            # Range-limited histogram
                            pre_warm, keep_alive = range_histogram_policy(func_class[func].IT_histogram)
                            func_class[func].window_param = (pre_warm, keep_alive)
                    elif IT >= RANGE and len(func_class[func].ITs) >=3:
                        if func not in arima_models or arima_models[func] is None:
                            if np.var(func_class[func].ITs) < 1e-6:  #ITs 的数据全为相同值,会导致 ARIMA 模型失效
                                next_it_prediction = func_class[func].ITs[0]                            
                            else:
                                try:
                                    model = pm.auto_arima(func_class[func].ITs, seasonal=True, stepwise=True, suppress_warnings=True)
                                    next_it_prediction = model.predict(n_periods=1)[0]
                                    arima_models[func] = model
                                except Exception as e:
                                    print(func_class[func].ITs)
                                    print(f"Error during model fitting: {e}")
                                    next_it_prediction = np.mean(func_class[func].ITs)

                        elif arima_models[func] is not None:
                            try:
                                model = arima_models[func]
                                model.update(np.array([IT]))
                                next_it_prediction = model.predict(n_periods=1)[0]
                            except Exception as e:
                                print(func_class[func].ITs)
                                print(f"Error during model fitting: {e}")
                                next_it_prediction = np.mean(func_class[func].ITs)
 
                        pre_warm = next_it_prediction * (1 - ARIMA_MARGIN)
                        keep_alive = next_it_prediction * ARIMA_MARGIN * 2
                        func_class[func].window_param = (pre_warm, keep_alive)
                    func_class[func].last_call = i
                    func_class[func].nearest_invoke = test_func_arrcount[func][i]

                else:  # not invoke
                    if func in memory:
                        waste_mem_time += func_class[func].containers_num
                        func_waste[func] += func_class[func].containers_num

                # 更新left_pre_warm和left_keep_alive
                if func_class[func].left_pre_warm > 0:
                    func_class[func].left_pre_warm -= 1
                elif func_class[func].left_keep_alive > 0:
                    func_class[func].left_keep_alive -= 1
                if test_func_arrcount[func][i] > 0:
                    func_class[func].left_pre_warm = pre_warm
                    func_class[func].left_keep_alive = keep_alive

    cold_ratio = []
    for func, cold in func_cold.items():
        if func_invok[func] !=0:
            cold_ratio.append(cold/func_invok[func])
        else:
            cold_ratio.append(0)
    print("WMT:", waste_mem_time/2880)
    print(f"CR_p50:{np.percentile(cold_ratio, 50)}\tCR_p75:{np.percentile(cold_ratio, 75)}\tCR_p90:{np.percentile(cold_ratio, 90)}\tCR_p95:{np.percentile(cold_ratio, 95)}")
    print("Total Cold Rate:", sum(func_cold.values())/sum(func_invok.values()))

    #保存记录
    os.makedirs(RESULT_DATA + "/HIST_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, RESULT_DATA + f"/HIST_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, RESULT_DATA + f"/HIST_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, RESULT_DATA + f"/HIST_result/func_invok_{cur_time}.json")

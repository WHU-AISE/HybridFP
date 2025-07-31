import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle as pkl
from numpy import fft
from common import *

HARMONICS = 10
LOCAL_WINDOW = 60

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = HARMONICS              # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

def icebreaker_predict(invoke_seq):
    input_seq = np.array(invoke_seq[-LOCAL_WINDOW:])
    n_predict = 1
    extrapolation = fourierExtrapolation(input_seq, n_predict)
    pred_value=extrapolation[-1]
    if pred_value <0:
        pred_value=0
    else:
        pred_value=round(pred_value)
    return pred_value    

if __name__ == "__main__": 
    MID_DATA = "../mid_data/AzureData"
    RESULT_DATA = "../result/AzureData"

    # 读取测试集数据并加载函数必要信息
    with open(MID_DATA + "/arrcount.pkl", "rb") as file:
        train_func_arrcount, test_func_arrcount = pkl.load(file)

    func_class = {}
    for func in test_func_arrcount:
        func_class[func] = func_state()
    print(len(func_class))

    memory = set()
    test_func = set(test_func_arrcount.keys())
    train_test_funcs = set(train_func_arrcount.keys()) & test_func
    pred_func_account = {func: [] for func in test_func}
    # 根据预测结果进行预热
    for func in train_test_funcs: # Pre_warm for testing at time 0
        arr = train_func_arrcount[func]
        pred_result = icebreaker_predict(arr)
        if pred_result > 0:
            func_class[func].set_containers(0, pred_result)
            pred_func_account[func].append(pred_result)
        else:
            pred_func_account[func].append(0)

    # 结果记录      
    func_cold = {func: 0 for func in test_func}    #冷启动次数
    func_invok = {func: 0 for func in test_func}   #调用次数
    func_invok_seq = {func:[] for func in test_func}

    func_invok_bool = {func: 0 for func in test_func}  #存在调用的单位时间
    func_waste = {func: 0 for func in test_func}   #内存浪费的单位时间   

    waste_mem_time = 0

    # 模拟
    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            pbar.update(1)                        
            
            for func in test_func: #In case of some functions staying in the memory forever
                # 函数状态更新
                func_class[func].setup()
                if func_class[func].state:
                    memory.add(func)
                else:
                    memory.remove(func) if func in memory else None
                
                if func in test_func_arrcount and test_func_arrcount[func][i] > 0:
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_seq[func].append(test_func_arrcount[func][i])
                    func_invok_bool[func] += 1

                    if not func_class[func].state: #cold start
                        func_cold[func] += test_func_arrcount[func][i]
                        func_class[func].cold_containers_update(test_func_arrcount[func][i])
                        memory.add(func)
                    elif func_class[func].state and func_class[func].containers_num < test_func_arrcount[func][i]: # 部分冷启动
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num
                        func_class[func].cold_containers_update(test_func_arrcount[func][i] - func_class[func].containers_num)
                    elif func_class[func].state and func_class[func].containers_num >= test_func_arrcount[func][i]: # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]
                        func_waste[func] += func_class[func].containers_num - test_func_arrcount[func][i]
                        func_class[func].cold_containers_update(0)     
                        
                else:   # not invoke
                    if func in memory:
                        func_class[func].cold_containers_update(0)
                        waste_mem_time += func_class[func].containers_num
                        if func in func_waste:
                            func_waste[func] += func_class[func].containers_num

                    if func_class[func].wait_time is None:
                        func_class[func].wait_time = 1
                    else:
                        func_class[func].wait_time += 1
                
                if func in train_func_arrcount:
                    train_arr = train_func_arrcount[func]
                    invoke_arr = train_func_arrcount[func]+func_invok_seq[func]
                    pred_result = icebreaker_predict(invoke_arr)
                    func_class[func].set_containers(i+1, pred_result)
                    pred_func_account[func].append(pred_result) if pred_result > 0 else pred_func_account[func].append(0)
                elif len(func_invok_seq[func]) >= LOCAL_WINDOW:
                    invoke_arr = func_invok_seq[func][-LOCAL_WINDOW:]
                    pred_result = icebreaker_predict(invoke_arr)
                    func_class[func].set_containers(i+1, pred_result)
                    pred_func_account[func].append(pred_result) if pred_result > 0 else pred_func_account[func].append(0)

    cold_ratio = []
    for func, cold in func_cold.items():
        if func_invok[func] != 0:
            cold_ratio.append(cold/func_invok[func])
        else:
            cold_ratio.append(0)
    print("WMT:", waste_mem_time/2880)
    print(f"CR_p50:{np.percentile(cold_ratio, 50)}\tCR_p75:{np.percentile(cold_ratio, 75)}\tCR_p90:{np.percentile(cold_ratio, 90)}\tCR_p95:{np.percentile(cold_ratio, 95)}")
    print("Total Cold Rate:", sum(func_cold.values())/sum(func_invok.values()))

    #保存记录
    os.makedirs(RESULT_DATA + "/IceBreaker_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, RESULT_DATA + f"/IceBreaker_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, RESULT_DATA + f"/IceBreaker_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, RESULT_DATA + f"/IceBreaker_result/func_invok_{cur_time}.json")
                                                  
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle as pkl
from common import *

KEEP_ALIVE = 10

if __name__ == "__main__": 
    MID_DATA = "../mid_data/AzureData"
    RESULT_DATA = "../result/AzureData"

    # 读取测试集数据并加载函数必要信息
    with open(MID_DATA+"/arrcount.pkl", "rb") as file:
        train_func_arrcount, test_func_arrcount = pkl.load(file)

    func_class = {}
    for func in train_func_arrcount:
        func_class[func] = func_state()
    print(len(func_class))

    # 加载测试集函数
    num_unseen_func = 0

    for func in test_func_arrcount:
        if func in func_class:
            continue
        num_unseen_func += 1
        func_class[func] = func_state()
        
    print(len(func_class), len(train_func_arrcount), len(test_func_arrcount), num_unseen_func)

    test_func = set(test_func_arrcount.keys())

    # 结果记录      
    func_cold = {func: 0 for func in test_func}    #冷启动次数
    func_invok = {func: 0 for func in test_func}   #调用次数

    func_invok_bool = {func: 0 for func in test_func}  #存在调用的单位时间
    func_waste = {func: 0 for func in test_func}   #内存浪费的单位时间   
    print(len(func_cold)) 
    waste_mem_time = 0
    
    train_test_funcs = set(train_func_arrcount.keys()) & test_func

    for func in train_test_funcs:
        train_arr = train_func_arrcount[func]
        invok = conj_seq_lst(train_arr, count_invoke=True)
        if(len(invok)==0):
            func_class[func].last_call = 0 - len(train_arr)
        else:
            func_class[func].last_call = np.where(np.array(train_arr)>0)[0][-1] - len(train_arr)
        if func_class[func].last_call + KEEP_ALIVE >= 0:
            # 上次调用后持续保活至此
            last_invoke = train_func_arrcount[func][np.where(np.array(train_func_arrcount[func])>0)[0][-1]]
            func_class[func].load(0, last_invoke)
            func_class[func].left_keep_alive = func_class[func].last_call + KEEP_ALIVE + 1
            func_class[func].instances.extend([func_class[func].last_call] * last_invoke)

    # 模拟
    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            if i % 60 == 59:
                pbar.update(60)
            for func in test_func:
                # 清理过期实例（超过 10 分钟未被复用）
                while func_class[func].instances and func_class[func].instances[0] + KEEP_ALIVE < i:
                    func_class[func].instances.popleft()                
                # 记录当前实例数
                func_class[func].containers_num = len(func_class[func].instances)

                if func in test_func_arrcount and test_func_arrcount[func][i] > 0:
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_bool[func] += 1
                    
                    # 根据冷启动情况记录冷启动数据和内存浪费数据
                    if func_class[func].containers_num < test_func_arrcount[func][i]:
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num      
                    elif func_class[func].containers_num >= test_func_arrcount[func][i]: # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]
                        func_waste[func] += func_class[func].containers_num - test_func_arrcount[func][i]

                    reuse_count = min(func_class[func].containers_num, test_func_arrcount[func][i])
                    # 更新复用实例的最后使用时间
                    for j in range(len(func_class[func].instances) - reuse_count, len(func_class[func].instances)):
                        func_class[func].instances[j] = i
                    
                    # 分配新实例（如果请求数大于可用实例数）
                    new_count = test_func_arrcount[func][i] - reuse_count
                    func_class[func].instances.extend([i] * new_count)                         

                else:      # not invoke
                    waste_mem_time += func_class[func].containers_num
                    func_waste[func] += func_class[func].containers_num                        


    # cold_ratio = [cold/func_invok[func] for func, cold in func_cold.items()]
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
    os.makedirs(RESULT_DATA + "/OW_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, RESULT_DATA + f"/OW_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, RESULT_DATA + f"/OW_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, RESULT_DATA + f"/OW_result/func_invok_{cur_time}.json")

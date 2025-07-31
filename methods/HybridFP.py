from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
import time
import sys
import polars as pl
import pickle as pkl
from collections import deque
from common import *

KEEP_ALIVE = 10

if __name__ == "__main__": 
    MID_DATA = "../mid_data/AzureData"
    RESULT_DATA = "../result/AzureData"
    # 参数设置
    LOCAL_WINDOW = 30
    PREDICT_WINDOW = 1
    with open(MID_DATA + "/arrcount.pkl", "rb") as file:
        train_func_arrcount, test_func_arrcount = pkl.load(file)

    func_class = {}
    for func in train_func_arrcount:
        func_class[func] = func_state()
    # 加载测试集函数
    num_unseen_func = 0

    for func in test_func_arrcount:
        if func in func_class:
            continue
        num_unseen_func += 1
        func_class[func] = func_state()
    print(len(func_class), len(train_func_arrcount), len(test_func_arrcount), num_unseen_func)

    # 过滤得可预测函数
    df = pl.read_csv(MID_DATA + "/func_info.csv")
    pe_df = df.filter(pl.col('PE') > 0.7)      
    df_union = pl.concat([pe_df]).unique()
    predictable_func_ids = df_union.select('Function').to_numpy().flatten().tolist()
    predictable_func_ids = list(map(lambda x: str(x), predictable_func_ids))
    predictable_func_ids = set(predictable_func_ids)

    pred_func_account = {}
    boosting_params = {
                        "objective": "regression",
                        "metric": "mape",
                        "verbosity": -1,
                        "boosting_type": "gbdt",
                        "seed": 42,
                        "learning_rate": 0.1,
                        "min_child_samples": 4,
                        "num_leaves": 128,
                        "num_iterations": 100
                                    }
    with tqdm(total=len(predictable_func_ids)) as pbar:
        for func in predictable_func_ids:
            lp_model = lp.LazyProphet(scale=True,
                            seasonal_period=[24, 168],
                            n_basis=8,
                            fourier_order=10,
                            ar=list(range(1, 97)),
                            decay=.99,
                            linear_trend=None,
                            decay_average=False,
                            boosting_params=boosting_params
                            )
            
            arr = train_func_arrcount[func]
            window_data = arr[-LOCAL_WINDOW:]
            lp_model.fit(window_data)            
            pred_result = lp_model.predict(PREDICT_WINDOW).flatten()
            pred_result = list(map(lambda x: round(x) if x > 0 else 0, pred_result))
            pred_func_account[func] = pred_result
            pbar.update(1)

    test_func = set(test_func_arrcount.keys())
    predictable_test_func = predictable_func_ids & test_func     #测试集可预测函数 
    with open(MID_DATA + '/other_func_periods.pkl', 'rb') as file:
        other_func_periods = pkl.load(file) 

    with open(MID_DATA + '/other_func_motif_nocover.pkl', 'rb') as file:
        other_func_motif = pkl.load(file)   

    print(f"Total Func Num: {len(test_func)}")
    print(f"Predictable Func Num: {len(predictable_test_func)}")
    print(f"Other Func Num: {len(other_func_periods)}")

    train_test_funcs = set(train_func_arrcount.keys()) & test_func
    online_buffers = {}
    func_cover_motif = {}
    #预热
    for func in train_test_funcs: # 有历史数据的函数
        # 根据预测结果进行预热
        if func in predictable_test_func:
            pred_result = pred_func_account[func]
            if pred_result[0] > 0:
                func_class[func].set_containers(0, pred_result[0])
        elif func in other_func_motif:
            motif_idx, neighbor_idx, best_distance, m = other_func_motif[func]
            motif_data = train_func_arrcount[func][motif_idx: motif_idx+m]
            if motif_idx%1440==0 or neighbor_idx%1440==0:
                func_cover_motif[func] = True     #直接延续使用模板
                func_class[func].set_containers(0, motif_data[0])
            else:
                func_cover_motif[func] = False    #采用MASS快速匹配
                if len(other_func_periods[func])==0:
                    online_buffers[func] = deque(maxlen=60)
                    online_buffers[func].extend((train_func_arrcount[func][-60:]))
                else:
                    window_len = min(int(other_func_periods[func][0])*2, 720)
                    online_buffers[func] = deque(maxlen=window_len)
                    online_buffers[func].extend(train_func_arrcount[func][-window_len:])                    

                start_idx = mass_match(online_buffers[func],motif_data) #模板中最匹配的子序列开始序号
                func_class[func].set_containers(0, motif_data[(start_idx+len(online_buffers[func])) % len(motif_data)])

    # 结果记录      
    func_cold = {func: 0 for func in test_func}    #冷启动次数
    func_invok = {func: 0 for func in test_func}   #调用次数
    func_invok_seq = {func:[] for func in test_func}

    func_invok_bool = {func: 0 for func in test_func}  #存在调用的单位时间
    func_waste = {func: 0 for func in test_func}   #内存浪费的单位时间
    print(len(func_cold))   
    waste_mem_time = 0

    # 模拟
    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            # # 每隔PREDICT_WINDOW分钟预测一次
            if i % PREDICT_WINDOW == PREDICT_WINDOW - 1 and i != 0 and i != 2879:
                tasks = [(func, train_func_arrcount[func], func_invok_seq[func], LOCAL_WINDOW, PREDICT_WINDOW)
                        for func in predictable_func_ids if func in test_func_arrcount]                
                with Pool(2) as pool:
                    results = pool.map_async(predict_func, tasks)                    
                    pool.close()
                    pool.join()
                try:
                    all_results = results.get()
                    for func, pred_result in all_results:
                        if func is not None:
                            pred_func_account[func] = pred_result                  
                except Exception as e:
                    print(f"Error in proccessing: {e}")             
            pbar.update(1)
            for func in test_func: #In case of some functions staying in the memory forever

                # 清理过期实例
                while func_class[func].instances and func_class[func].instances[0] + KEEP_ALIVE < i:
                    func_class[func].instances.popleft() 
                func_class[func].set_up()
                
                if func in test_func_arrcount and test_func_arrcount[func][i] > 0:
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_seq[func].append(test_func_arrcount[func][i])
                    func_invok_bool[func] += 1
                    
                    if func_class[func].containers_num < test_func_arrcount[func][i]: # 冷启动
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num
                    elif func_class[func].containers_num >= test_func_arrcount[func][i]: # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]
                        func_waste[func] += func_class[func].containers_num - test_func_arrcount[func][i]

                    reuse_count = min(len(func_class[func].instances), test_func_arrcount[func][i]-func_class[func].controlled_containers_num)
                    # 更新复用实例的最后使用时间
                    for j in range(len(func_class[func].instances) - reuse_count, len(func_class[func].instances)):
                        func_class[func].instances[j] = i
                    # 分配新实例（如果请求数大于可用实例数）
                    new_count = test_func_arrcount[func][i] - func_class[func].controlled_containers_num - reuse_count
                    func_class[func].instances.extend([i] * new_count)                       

                else:   # not invoke
                    waste_mem_time += func_class[func].containers_num
                    if func in func_waste:
                        func_waste[func] += func_class[func].containers_num
                
                if func in predictable_test_func:     #可预测函数 
                    pred_result = pred_func_account[func][(i+1) % PREDICT_WINDOW]
                    # pred_result = pred_func_account[func][i+1]  if i < 2879 else pred_func_account[func][0]
                    func_class[func].set_containers(i+1, pred_result)
                elif func in other_func_motif:
                    motif_idx, neighbor_idx, best_distance, m = other_func_motif[func]
                    origin_data = train_func_arrcount[func][-1440*3:]
                    motif_data = origin_data[motif_idx: motif_idx+m]
                    if func_cover_motif[func]==True:
                        func_class[func].set_containers(i+1, motif_data[(i+1) % 1440])
                    elif func_cover_motif[func]==False:
                        online_buffers[func].append(test_func_arrcount[func][i])
                        time1 = time.time()
                        start_idx = mass_match(online_buffers[func], motif_data) #模板中最匹配的子序列开始序号
                        time2 = time.time()
                        print(f"Time taken: {time2 - time1} seconds")
                        func_class[func].set_containers(i+1, motif_data[(start_idx+len(online_buffers[func])) % len(motif_data)])

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
    os.makedirs(RESULT_DATA + "/HybridFP_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, RESULT_DATA + f"/HybridFP_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, RESULT_DATA + f"/HybridFP_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, RESULT_DATA + f"/HybridFP_result/func_invok_{cur_time}.json")
                                                   
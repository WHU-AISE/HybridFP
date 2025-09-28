import os 
import json
import numpy as np
import time
from math import factorial
from LazyProphet import LazyProphet as lp
from scipy.stats import expon
import math
import stumpy
from collections import deque

FILL_ZERO_FLAG = True
shown_func_num = 1000
UNKNOWN, WARM, REGULAR, APPRO_REGULAR, DENSE, SUCCESSIVE, PLUSED, POSSIBLE, CORR, NEW_POSS = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 #Diviable? Active?
TYPE_NUM = 10

#PESE PARAMETER
SEQ_NUM_BOUND = 3 #Too infrequently invoked functions should not be categorized to this type.
IDLE_NUM_MAX = 3 #Consider the first n modes
IDLE_PERCEN = 0.9
DENSE_UPPER_BOUND = 5 #the small constant
DENSE_PERCEN = 90
PLUSED_GIVEUP = 3
POSS_GIVEUP = 1
CORR_GIVEUP = 1
PRE_WARM = 2
DISCRETE_TH = 10
GIVE_UP = {
    UNKNOWN: POSS_GIVEUP,
    WARM: 1440 * 14,
    REGULAR: 1,
    APPRO_REGULAR: 1,
    DENSE: DENSE_UPPER_BOUND,
    SUCCESSIVE: 1,
    PLUSED: PLUSED_GIVEUP,
    POSSIBLE: POSS_GIVEUP,
    CORR: CORR_GIVEUP,
    NEW_POSS: POSS_GIVEUP,
}
CORR_REMOVAL_TH = 2
EN_STD = 2


SHIFT = True #adaptive strategies
CONCURRENCY = False  #Concurrent execution 

label_lst = ['Unknown','Warm', 'Regular', "Appro-regular", "Dense", "Successive", "Plused", "Possible", "Corr"]
color_lst = ['#F5B3A2','#BC90B6','#2A4597','#DDC3C7', '#6A9C79', '#C5563B',"#F1AB3D", '#58A0A4','#E5F1E5',"#FBDEAF"]

class func_state:
    def __init__(self, _type = 0, forget = 0):
        self.type = _type
        self.forget = forget
        
        self.state = False # loaded or not
        self.load_time = None 
        self.wait_time = None 
        self.last_call = None
        self.pre_call_start = None # start of the last calling series
        
        self.idle_info = {} # "mode"：WT mode、 "mode_count": mode 出现次数
        self.invok_info = {}
        self.lasting_info = {}  #

        self.pred_interval = [] # 预测值
        self.pred_value = []
        self.next_invok_start = []
        
        self.adp_wait = []
        self.containers_num = 0
        self.instances = deque()    # 记录当前每个实例的最后使用时间

        # AMC
        self.containers_dict = {}
        self.cold_containers = []
        self.controlled_containers_num = 0

        # HIST
        self.ITs = []
        self.IT_histogram = [0] * 240
        self.window_param = None
        self.left_pre_warm = 0
        self.left_keep_alive = 0
        self.nearest_invoke = None
        
        
    def load(self, load_time, load_num=None):
        self.state = True
        self.load_time = load_time
        if load_num != None:
            self.containers_num += load_num
    
    def cal_lasting(self, cur_time):
        if not self.state:
            return 0
        return cur_time - self.load_time + 1
    
    def unload(self):
        self.state = False
        self.load_time = None
        self.containers_num = 0
        
    def set_containers(self, set_time, containers_num):
        self.load_time = set_time
        self.controlled_containers_num = containers_num

    def cal_wait(self):
        if self.wait_time is None:
            self.wait_time = 0
        self.wait_time += 1
    
    def reset(self, pred=False):
        self.unload()
        self.wait_time = None 
        self.last_call = None
        self.pre_call_start = None
        
        self.adp_wait = []
        
        if pred:
            self.next_invok_start = []
    
    # AMC
    def cold_containers_update(self, cold_num):
        if len(self.cold_containers) == 10:
            self.cold_containers.pop(0)
        self.cold_containers.append(cold_num)        
    
    # AMC
    def setup(self):
        self.containers_num = self.controlled_containers_num + sum(self.cold_containers)
        if self.containers_num > 0:
            self.state = True
        else:
            self.state = False
            self.load_time = None
    
    def set_up(self):
        self.containers_num = self.controlled_containers_num + len(self.instances)
        if self.containers_num > 0:
            self.state = True
        else:
            self.state = False
            self.load_time = None



class distribution:
    def __init__(self, history_timeout, history_maxlength):
        self.history_timeout = history_timeout
        self.history_maxlength = history_maxlength
        self.history = []
    
    def update(self, arrival_time):
        self.history = list(filter(lambda x: arrival_time - x < self.history_timeout, self.history))
        if len(self.history) == self.history_maxlength:
            self.history.pop(0)

        self.history.append(arrival_time)

    def predict_IAT(self, min, current_time, quantile):
        if len(self.history) == 0:
            return min
        else:
            mean = (current_time - self.history[0]) / len(self.history)
            if mean > 0:
                exponential = expon(scale=mean)
                IAT = math.ceil(exponential.ppf(quantile))
                return IAT
            else:
                return min

def mass_match(online_buffer, motif_data):
    buffer_data = np.array(online_buffer).astype(float)
    motif_data = np.array(motif_data).astype(float)
    distance_profile = stumpy.mass(buffer_data, motif_data)       
    idx = np.argmin(distance_profile)
    return idx

# 求连续调用序列或连续未调用序列 串
def conj_seq_lst(lst, count_invoke=False, threshold=1):
    seq_lenth_lst = []
    pre_pos = -1
    for i, e in enumerate(lst):
        if not (bool(e) ^ count_invoke): #非异或 判断两个条件是否相等 (求连续正且当前元素为正 或 求连续负且当前元素为负)
            if pre_pos < 0:     #连续序列中的第一个元素位置
                pre_pos = i
            if i == len(lst)-1 and i+1-pre_pos >= threshold:    #末尾元素进行处理, 且连续序列长度大于阈值
                seq_lenth_lst.append(i+1-pre_pos)
        else:   # 连续序列中断
            if pre_pos>=0 and i-pre_pos >= threshold:
                seq_lenth_lst.append(i-pre_pos)
            pre_pos = -1
    return seq_lenth_lst

def conj_local_invok(lst, gap_tolerance=5):
    # 初始化变量
    in_burst = False  # 是否处于突发状态
    burst_start = None
    seq_lenth_lst = []

    # 遍历数据
    for i, value in enumerate(lst):
        if value > 0:  
            if not in_burst:
                burst_start = i  # 记录突发段起始点
                in_burst = True
            zero_counter = 0  # 重置零值计数器
        elif in_burst and value == 0:  # 在突发段内遇到零值
            zero_counter += 1
            if zero_counter >= gap_tolerance:  # 连续零值超过最大容忍值
                seq_lenth_lst.append(i - burst_start - zero_counter + 1)  # 记录突发段长度
                in_burst = False  # 退出突发状态
                burst_start = None
        elif in_burst and value <= 0:  # 遇到非突发值
            zero_counter = 0  # 重置零值计数器

    # 处理最后的突发段
    if in_burst:
        seq_lenth_lst.append(len(lst) - burst_start)
    return seq_lenth_lst

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        print("File path "+filepath+" not exists!")
        return
    
def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)

def add_value_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha='center', va='bottom')

def _embed(x, order=3, delay=1):
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

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

# MAE (平均绝对误差)
def mae(y_true, y_pred):
    result = np.mean(np.abs(y_true - y_pred))
    return round(result,3)

# MSE (均方误差)
def mse(y_true, y_pred):
    result = np.mean((y_true - y_pred) ** 2)
    return round(result, 3)

# sMAPE (对称平均绝对百分比误差)
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_value = np.mean(np.abs(y_pred - y_true) / (denominator + 1e-10))
    return round(smape_value, 3)

# RMSE (均方根误差)
def rmse(y_true, y_pred):
    result = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return round(result, 3)

def sliding_window_prediction(model, train_data, valid_data, local_window, predict_window):
    valid_size = len(valid_data)
    predictions = []
    extended_train_data = np.copy(train_data)
    total_time = 0  # 用于记录总时间
    prediction_count = 0  # 记录预测的次数

    for start in range(0, valid_size, predict_window):
        window_data = np.concatenate((extended_train_data, valid_data[:start]))

        if len(window_data) > local_window:
            window_data = window_data[-local_window:]

        start_time = time.time()  
        model.fit(window_data)
        pred = model.predict(predict_window).flatten()
        end_time = time.time()  

        # 本次预测用时
        prediction_time = end_time - start_time
        total_time += prediction_time
        prediction_count += 1

        pred = list(map(lambda x: round(x) if x > 0 else 0, pred))
        predictions.extend(pred)
    #平均用时
    average_time = total_time / prediction_count if prediction_count > 0 else 0
    return predictions, average_time


#   LazyProphet 拟合预测函数负载
def predict_func(args):
    func, arr, invok_seq, LOCAL_WINDOW, PREDICT_WINDOW = args
    # LighGBM 的参数
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
    window_data = np.concatenate((arr, invok_seq))
    if len(window_data) > LOCAL_WINDOW:
        window_data = window_data[-LOCAL_WINDOW:]
    lp_model.fit(window_data)
    pred_result = lp_model.predict(PREDICT_WINDOW).flatten()
    pred_result = list(map(lambda x: round(x) if x > 0 else 0, pred_result))
    
    return func, pred_result

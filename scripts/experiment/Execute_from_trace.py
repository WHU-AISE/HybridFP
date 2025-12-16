import math
import pickle
import os
import time
import pandas as pd
import random
import json
import sys
import asyncio
import redis
from concurrent.futures import ThreadPoolExecutor


sys.path.append('..')
sys.path.append('../..')
from params import WSK_CLI, COUCH_LINK
from run_cmd import run_cmd
from logger import Logger

# 读取 redis 配置信息
REDIS_HOST = "172.17.0.1"
REDIS_PORT = 6379
REDIS_PASSWORD = "openwhisk"
NAMESPACE="guest/"

# Connect to redis pool
def get_redis(
    redis_host, 
    redis_port, 
    redis_password
):
    pool = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    redis_client = redis.Redis(connection_pool=pool)

    return redis_client

# 异步调用函数
async def invoke_async(function_id, param):
    """异步调用函数并返回时序指标"""
    loop = asyncio.get_event_loop()
    
    # 使用线程池执行阻塞的I/O操作
    with ThreadPoolExecutor() as executor:
        try:
            cmd = '{} action invoke {} -b {} | grep -v "ok:"'.format(WSK_CLI, function_id, param)
            response = await loop.run_in_executor(executor, run_cmd, cmd)
            doc = json.loads(response)

            if len(doc["annotations"]) == 6:
                is_cold_start = True
                init_time = doc["annotations"][5]["value"]
            else:
                init_time = 0
                is_cold_start = False
            wait_time = doc["annotations"][1]["value"]
            duration = doc["duration"]
        except Exception as e:
            print(f"Error invoking {function_id}: {e}")
            init_time = -1
            wait_time = -1
            duration = -1
            is_cold_start = False
    
    return is_cold_start, init_time, wait_time, duration

# Load selected functions and their invocation counts from pickle files
with open('selected_functions_240min.pkl', 'rb') as f:
    selected_functions = pickle.load(f)

with open('selected_function_invocations_240min.pkl', 'rb') as f:
    selected_function_invocations = pickle.load(f)

selected_functions_list = [selected_functions.iloc[i] for i in range(16)]

# Get invocation counts for these selected functions
invocations_list = [selected_function_invocations[func["HashFunction"]] for func in selected_functions_list]

function_name_list = ["ac", "dl", "is", "ds", "dh", "dt", "dq", "dg", "fc", "dv", "gm", "gb", "oi", "sa", "tn", "md"]


# Initialize logger for recording metrics
logger_wrapper = Logger()
logger = logger_wrapper.get_logger("trace_execution_metrics_480min", True)

logger.debug("Starting trace execution with metrics recording")
logger.debug("Function list: " + ','.join([func["HashFunction"] for func in selected_functions_list]))

def generate_randomized_schedule(count, minute_duration=60, randomness=0.3):
    """
    为指定次数的调用生成带有随机性的时间调度
    
    Args:
        count: 调用次数
        minute_duration: 分钟时长（秒）
        randomness: 随机性程度（0-1），0表示完全均匀，1表示完全随机
    """
    if count == 0:
        return []
    
    if count == 1:
        # 单次调用放在分钟中间
        base_time = minute_duration / 2
        random_offset = (random.random() - 0.5) * minute_duration * randomness
        return [max(0, min(minute_duration - 0.1, base_time + random_offset))]
    
    # 生成基础均匀分布
    base_times = [i * (minute_duration / count) for i in range(count)]
    
    # 添加随机偏移
    scheduled_times = []
    max_offset = (minute_duration / count) * randomness
    
    for base_time in base_times:
        random_offset = (random.random() - 0.5) * 2 * max_offset
        scheduled_time = base_time + random_offset
        # 确保时间在有效范围内
        scheduled_time = max(0, min(minute_duration - 0.1, scheduled_time))
        scheduled_times.append(scheduled_time)
    
    # 重新排序以确保时间顺序
    scheduled_times.sort()
    return scheduled_times

async def execute_and_log_invocation(function_name, param, minute, scheduled_time, actual_time):
    """执行单个函数调用并记录结果（不阻塞主循环）"""
    try:
        is_cold_start, init_time, wait_time, duration = await invoke_async(function_name, param)
        
        # 记录完成时间
        completion_time = time.time()
        
        # Log the metrics
        logger.debug(f"function_name: {function_name}")
        logger.debug(f"minute: {minute}, scheduled_sec: {scheduled_time:.2f}, actual_start_sec: {actual_time:.2f}, completion_sec: {completion_time:.2f}")
        logger.debug(f"is_cold_start: {is_cold_start}")
        logger.debug(f"init_time: {init_time}")
        logger.debug(f"wait_time: {wait_time}")
        logger.debug(f"duration: {duration}")
        logger.debug("--------------------------------")
        
    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        # 记录错误信息
        logger.debug(f"function_name: {function_name}")
        logger.debug(f"minute: {minute}, scheduled_sec: {scheduled_time:.2f}, actual_start_sec: {actual_time:.2f}")
        logger.debug(f"ERROR: {str(e)}")
        logger.debug("--------------------------------")

# 异步主循环
async def main_async():
    """异步主循环 - 创建任务但不等待完成"""
    redis_client = get_redis(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)

    # 记录实验开始时间
    experiment_start_time = time.time()
    # Simulate function invocation over a period of 480 minutes
    for minute in range(1, 241):
        
        logger.debug(f"\n=== Starting minute {minute} ===")
        # 为每个函数生成本分钟的调用时间表
        schedules = []
        for i in range(16):
            count = invocations_list[i].get(str(minute), 0)
            if count > 0:
                scheduled_times = generate_randomized_schedule(count)
                for scheduled_time in scheduled_times:
                    schedules.append((scheduled_time, i))
        
        # 按时间排序所有调用
        schedules.sort(key=lambda x: x[0])
        
        minute_start_time = time.time()
        
        # 为每个调用创建异步任务
        tasks = []
        for scheduled_time, func_index in schedules:
            # 计算需要等待的时间
            current_elapsed = time.time() - minute_start_time
            wait_time = scheduled_time - current_elapsed
            
            # 如果已经超过预定时间，立即执行；否则等待
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # 计算实际调用时间
            actual_time = time.time() - minute_start_time
            absolute_actual_time = time.time() - experiment_start_time
            
            function_name = function_name_list[func_index]
            param = f"-p couch_link {COUCH_LINK} -p db_name {function_name}"
            
            # 创建异步任务但不等待完成
            task = asyncio.create_task(
                execute_and_log_invocation(
                    function_name, param, minute, scheduled_time, actual_time
                )
            )
            tasks.append(task)
            
        # 确保每分钟完整执行60秒
        minute_elapsed = time.time() - minute_start_time
        if minute_elapsed < 60:
            await asyncio.sleep(60 - minute_elapsed)
        
        logger.debug(f"Minute {minute}: Created {len(tasks)} invocation tasks")
        
        # 注意：这里不等待任务完成，继续下一分钟的执行
        # 任务会在后台自动执行和完成

# 运行异步主函数
if __name__ == "__main__":
    logger.debug("Starting asynchronous trace execution with metrics recording")
    
    # 运行异步主循环
    asyncio.run(main_async())
    
    logger.debug("\nTrace execution with metrics recording completed!")
    print("")
    print("Trace execution with metrics recording finished!")
    print("")
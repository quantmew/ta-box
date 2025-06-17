import numpy as np
import time
from tabox.ta_func.ta_MACDEXT import MACDEXT

def bench_macdext():
    # 生成测试数据
    data = np.random.random(10000)
    
    # 测试不同参数组合
    params = [
        (12, 0, 26, 0, 9, 0),  # 默认参数
        (5, 0, 10, 0, 3, 0),   # 短周期
        (20, 0, 40, 0, 15, 0), # 长周期
        (12, 1, 26, 1, 9, 1),  # 使用EMA
        (12, 2, 26, 2, 9, 2),  # 使用WMA
    ]
    
    print("MACDEXT Benchmark Results:")
    print("-" * 50)
    
    for fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype in params:
        # 预热
        for _ in range(3):
            MACDEXT(data, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)
        
        # 计时
        start_time = time.time()
        for _ in range(10):
            MACDEXT(data, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)
        end_time = time.time()
        
        # 计算平均时间
        avg_time = (end_time - start_time) / 10
        
        print(f"Parameters: fastperiod={fastperiod}, fastmatype={fastmatype}, "
              f"slowperiod={slowperiod}, slowmatype={slowmatype}, "
              f"signalperiod={signalperiod}, signalmatype={signalmatype}")
        print(f"Average time: {avg_time:.6f} seconds")
        print("-" * 50)

if __name__ == "__main__":
    bench_macdext() 
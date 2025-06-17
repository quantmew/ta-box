from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import numpy as np
from tabox import RSI as this_RSI
from talib import RSI as that_RSI

@bench
def bench_this_rsi():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_RSI(close, timeperiod = 30)

@bench
def bench_that_rsi():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_RSI(close, timeperiod = 30)

if __name__ == '__main__':
    bench_this_rsi()
    bench_that_rsi()

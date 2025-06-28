from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ATR as this_ATR
from talib import ATR as that_ATR

@bench
def bench_this_atr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_ATR(high, low, close, timeperiod=t)

@bench
def bench_that_atr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_ATR(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_atr()
    bench_that_atr() 
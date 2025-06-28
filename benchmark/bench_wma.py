from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import WMA as this_WMA
from talib import WMA as that_WMA

@bench
def bench_this_wma():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_WMA(close, timeperiod=t)

@bench
def bench_that_wma():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_WMA(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_wma()
    bench_that_wma() 
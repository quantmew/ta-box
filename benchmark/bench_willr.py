from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import WILLR as this_WILLR
from talib import WILLR as that_WILLR

@bench
def bench_this_willr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_WILLR(high, low, close, timeperiod=t)

@bench
def bench_that_willr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_WILLR(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_willr()
    bench_that_willr() 
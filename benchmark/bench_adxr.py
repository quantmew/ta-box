from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ADXR as this_ADXR
from talib import ADXR as that_ADXR

@bench
def bench_this_adxr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_ADXR(high, low, close, timeperiod=t)

@bench
def bench_that_adxr():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_ADXR(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_adxr()
    bench_that_adxr() 
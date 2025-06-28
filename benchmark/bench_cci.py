from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import CCI as this_CCI
from talib import CCI as that_CCI

@bench
def bench_this_cci():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_CCI(high, low, close, timeperiod=t)

@bench
def bench_that_cci():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_CCI(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_cci()
    bench_that_cci() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MA as this_MA
from talib import MA as that_MA

@bench
def bench_this_ma():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_MA(close, timeperiod=t)

@bench
def bench_that_ma():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_MA(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_ma()
    bench_that_ma() 
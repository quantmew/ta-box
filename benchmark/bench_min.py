from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MIN as this_MIN
from talib import MIN as that_MIN

@bench
def bench_this_min():
    close = np.arange(1000000, 10, -1, dtype=np.float64)
    this_ret = this_MIN(close, timeperiod=500)

    for i in range(10000, 10500):
        for t in [2, 3, 5, 10, 30, 50]:
            close = np.random.random(i)
            this_ret = this_MIN(close, timeperiod=t)

@bench
def bench_that_min():
    close = np.arange(1000000, 10, -1, dtype=np.float64)
    this_ret = that_MIN(close, timeperiod=500)

    for i in range(10000, 10500):
        for t in [2, 3, 5, 10, 30, 50]:
            close = np.random.random(i)
            that_ret = that_MIN(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_min()
    bench_that_min()

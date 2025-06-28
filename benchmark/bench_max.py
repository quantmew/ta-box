from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MAX as this_MAX
from talib import MAX as that_MAX

@bench
def bench_this_max():

    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_MAX(close, timeperiod=t)

@bench
def bench_that_max():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_MAX(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_max()
    bench_that_max()

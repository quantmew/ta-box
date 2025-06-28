from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ROCP as this_ROCP
from talib import ROCP as that_ROCP

@bench
def bench_this_rocp():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_ROCP(close, timeperiod=t)

@bench
def bench_that_rocp():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_ROCP(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_rocp()
    bench_that_rocp() 
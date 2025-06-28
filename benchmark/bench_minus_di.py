from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MINUS_DI as this_MINUS_DI
from talib import MINUS_DI as that_MINUS_DI

@bench
def bench_this_minus_di():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_MINUS_DI(high, low, close, timeperiod=t)

@bench
def bench_that_minus_di():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_MINUS_DI(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_minus_di()
    bench_that_minus_di() 
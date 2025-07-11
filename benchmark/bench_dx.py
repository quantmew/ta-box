from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import DX as this_DX
from talib import DX as that_DX

@bench
def bench_this_dx():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_DX(high, low, close, timeperiod=t)

@bench
def bench_that_dx():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_DX(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_dx()
    bench_that_dx() 
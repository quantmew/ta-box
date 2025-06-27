from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import NATR as this_NATR
from talib import NATR as that_NATR

@bench
def bench_this_natr():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_NATR(high, low, close, timeperiod=t)

@bench
def bench_that_natr():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            that_ret = that_NATR(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_natr()
    bench_that_natr() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ADX as this_ADX
from talib import ADX as that_ADX

@bench
def bench_this_adx():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_ADX(high, low, close, timeperiod=t)

@bench
def bench_that_adx():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            that_ret = that_ADX(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_adx()
    bench_that_adx() 
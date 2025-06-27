from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MIDPRICE as this_MIDPRICE
from talib import MIDPRICE as that_MIDPRICE

@bench
def bench_this_midprice():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            high = np.random.random(i)
            low = np.random.random(i)
            this_ret = this_MIDPRICE(high, low, timeperiod=t)

@bench
def bench_that_midprice():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            high = np.random.random(i)
            low = np.random.random(i)
            that_ret = that_MIDPRICE(high, low, timeperiod=t)

if __name__ == '__main__':
    bench_this_midprice()
    bench_that_midprice() 
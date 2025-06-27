from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import WCLPRICE as this_WCLPRICE
from talib import WCLPRICE as that_WCLPRICE

@bench
def bench_this_wclprice():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_WCLPRICE(high, low, close)

@bench
def bench_that_wclprice():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_WCLPRICE(high, low, close)

if __name__ == '__main__':
    bench_this_wclprice()
    bench_that_wclprice() 
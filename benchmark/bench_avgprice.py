from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import AVGPRICE as this_AVGPRICE
from talib import AVGPRICE as that_AVGPRICE

@bench
def bench_this_avgprice():
    for i in range(100, 2000):
        open_ = np.random.random(i)
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_AVGPRICE(open_, high, low, close)

@bench
def bench_that_avgprice():
    for i in range(100, 2000):
        open_ = np.random.random(i)
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_AVGPRICE(open_, high, low, close)

if __name__ == '__main__':
    bench_this_avgprice()
    bench_that_avgprice() 
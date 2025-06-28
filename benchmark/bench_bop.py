from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import BOP as this_BOP
from talib import BOP as that_BOP

@bench
def bench_this_bop():
    for i in range(100, 2000):
        open_ = np.random.random(i)
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_BOP(open_, high, low, close)

@bench
def bench_that_bop():
    for i in range(100, 2000):
        open_ = np.random.random(i)
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_BOP(open_, high, low, close)

if __name__ == '__main__':
    bench_this_bop()
    bench_that_bop() 
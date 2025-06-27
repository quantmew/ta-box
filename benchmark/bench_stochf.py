from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import STOCHF as this_STOCHF
from talib import STOCHF as that_STOCHF

@bench
def bench_this_stochf():
    for i in range(100, 2000):
        for fastk in [5, 10, 14]:
            for fastd in [3, 5, 10]:
                high = np.random.random(i)
                low = np.random.random(i)
                close = np.random.random(i)
                this_ret = this_STOCHF(high, low, close, fastk_period=fastk, fastd_period=fastd)

@bench
def bench_that_stochf():
    for i in range(100, 2000):
        for fastk in [5, 10, 14]:
            for fastd in [3, 5, 10]:
                high = np.random.random(i)
                low = np.random.random(i)
                close = np.random.random(i)
                that_ret = that_STOCHF(high, low, close, fastk_period=fastk, fastd_period=fastd)

if __name__ == '__main__':
    bench_this_stochf()
    bench_that_stochf() 
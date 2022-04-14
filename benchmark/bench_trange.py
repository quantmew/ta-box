from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TRANGE as this_TRANGE
from talib import TRANGE as that_TRANGE

@bench
def bench_this_trange():
    for i in range(100, 10000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_TRANGE(high, low, close)

@bench
def bench_that_trange():
    for i in range(100, 10000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_TRANGE(high, low, close)

if __name__ == '__main__':
    bench_this_trange()
    bench_that_trange()

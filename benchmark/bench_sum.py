from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SUM as this_SUM
from talib import SUM as that_SUM

@bench
def bench_this_sum():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_SUM(close, timeperiod=t)

@bench
def bench_that_sum():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_SUM(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_sum()
    bench_that_sum() 
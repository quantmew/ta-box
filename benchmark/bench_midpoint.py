from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MIDPOINT as this_MIDPOINT
from talib import MIDPOINT as that_MIDPOINT

@bench
def bench_this_midpoint():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_MIDPOINT(close, timeperiod=t)

@bench
def bench_that_midpoint():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_MIDPOINT(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_midpoint()
    bench_that_midpoint() 
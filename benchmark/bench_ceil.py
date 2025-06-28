from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import CEIL as this_CEIL
from talib import CEIL as that_CEIL

@bench
def bench_this_ceil():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_CEIL(close)

@bench
def bench_that_ceil():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_CEIL(close)

if __name__ == '__main__':
    bench_this_ceil()
    bench_that_ceil() 
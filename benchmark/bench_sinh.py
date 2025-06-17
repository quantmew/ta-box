from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SINH as this_SINH
from talib import SINH as that_SINH

@bench
def bench_this_sinh():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_SINH(close)

@bench
def bench_that_sinh():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_SINH(close)

if __name__ == '__main__':
    bench_this_sinh()
    bench_that_sinh() 
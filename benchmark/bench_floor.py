from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import FLOOR as this_FLOOR
from talib import FLOOR as that_FLOOR

@bench
def bench_this_floor():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_FLOOR(close)

@bench
def bench_that_floor():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_FLOOR(close)

if __name__ == '__main__':
    bench_this_floor()
    bench_that_floor() 
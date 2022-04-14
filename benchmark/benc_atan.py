from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ATAN as this_ATAN
from talib import ATAN as that_ATAN

@bench
def bench_this_atan():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_ATAN(close)

@bench
def bench_that_atan():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_ATAN(close)

if __name__ == '__main__':
    bench_this_atan()
    bench_that_atan()

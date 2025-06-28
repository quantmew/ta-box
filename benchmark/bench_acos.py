from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ACOS as this_ACOS
from talib import ACOS as that_ACOS

@bench
def bench_this_acos():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_ACOS(close)

@bench
def bench_that_acos():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_ACOS(close)

if __name__ == '__main__':
    bench_this_acos()
    bench_that_acos()

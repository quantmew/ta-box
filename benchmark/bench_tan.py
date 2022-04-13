from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TAN as this_TAN
from talib import TAN as that_TAN

@bench
def bench_this_tan():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_TAN(close)

@bench
def bench_that_tan():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_TAN(close)

if __name__ == '__main__':
    bench_this_tan()
    bench_that_tan()

from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import COSH as this_COSH
from talib import COSH as that_COSH

@bench
def bench_this_cosh():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_COSH(close)

@bench
def bench_that_cosh():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_COSH(close)

if __name__ == '__main__':
    bench_this_cosh()
    bench_that_cosh() 
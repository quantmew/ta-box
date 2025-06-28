from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ASIN as this_ASIN
from talib import ASIN as that_ASIN

@bench
def bench_this_asin():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_ASIN(close)

@bench
def bench_that_asin():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_ASIN(close)

if __name__ == '__main__':
    bench_this_asin()
    bench_that_asin()

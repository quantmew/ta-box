from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SQRT as this_SQRT
from talib import SQRT as that_SQRT

@bench
def bench_this_sqrt():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_SQRT(close)

@bench
def bench_that_sqrt():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_SQRT(close)

if __name__ == '__main__':
    bench_this_sqrt()
    bench_that_sqrt()

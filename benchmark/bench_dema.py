from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import DEMA as this_DEMA
from talib import DEMA as that_DEMA

@bench
def bench_this_dema():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_dema = this_DEMA(close)

@bench
def bench_that_dema():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_dema = that_DEMA(close)

if __name__ == '__main__':
    bench_this_dema()
    bench_that_dema() 
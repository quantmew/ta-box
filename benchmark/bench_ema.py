from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import EMA as this_EMA
from talib import EMA as that_EMA

@bench
def bench_this_ema():
    for i in range(100, 1000):
        close = np.random.random(i)
        this_ema = this_EMA(close)

@bench
def bench_that_ema():
    for i in range(100, 1000):
        close = np.random.random(i)
        that_ema = that_EMA(close)

if __name__ == '__main__':
    bench_this_ema()
    bench_that_ema() 
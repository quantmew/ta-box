from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ULTOSC as this_ULTOSC
from talib import ULTOSC as that_ULTOSC

@bench
def bench_this_ultosc():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_ULTOSC(high, low, close)

@bench
def bench_that_ultosc():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_ULTOSC(high, low, close)

if __name__ == '__main__':
    bench_this_ultosc()
    bench_that_ultosc() 
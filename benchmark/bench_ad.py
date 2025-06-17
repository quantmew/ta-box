from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import AD as this_AD
from talib import AD as that_AD

@bench
def bench_this_ad():
    for i in range(100, 10000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        volume = np.random.random(i)
        this_ret = this_AD(high, low, close, volume)

@bench
def bench_that_ad():
    for i in range(100, 10000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        volume = np.random.random(i)
        that_ret = that_AD(high, low, close, volume)

if __name__ == '__main__':
    bench_this_ad()
    bench_that_ad()
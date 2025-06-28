from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TYPPRICE as this_TYPPRICE
from talib import TYPPRICE as that_TYPPRICE

@bench
def bench_this_typprice():
    for i in range(100, 2000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        this_ret = this_TYPPRICE(high, low, close)

@bench
def bench_that_typprice():
    for i in range(100, 2000):
        high = np.random.random(i)
        low = np.random.random(i)
        close = np.random.random(i)
        that_ret = that_TYPPRICE(high, low, close)

if __name__ == '__main__':
    bench_this_typprice()
    bench_that_typprice() 
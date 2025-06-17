from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MACD as this_MACD
from talib import MACD as that_MACD

@bench
def bench_this_macd():
    for i in range(100, 1000):
        close = np.random.random(i)
        this_macd, this_signal, this_hist = this_MACD(close)

@bench
def bench_that_macd():
    for i in range(100, 1000):
        close = np.random.random(i)
        that_macd, that_signal, that_hist = that_MACD(close)

if __name__ == '__main__':
    bench_this_macd()
    bench_that_macd() 
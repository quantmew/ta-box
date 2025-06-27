from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TRIX as this_TRIX
from talib import TRIX as that_TRIX

@bench
def bench_this_trix():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_TRIX(close, timeperiod=t)

@bench
def bench_that_trix():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_TRIX(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_trix()
    bench_that_trix() 
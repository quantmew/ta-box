from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TSF as this_TSF
from talib import TSF as that_TSF

@bench
def bench_this_tsf():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_TSF(close, timeperiod=t)

@bench
def bench_that_tsf():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_TSF(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_tsf()
    bench_that_tsf() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import PLUS_DI as this_PLUS_DI
from talib import PLUS_DI as that_PLUS_DI

@bench
def bench_this_plus_di():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_PLUS_DI(high, low, close, timeperiod=t)

@bench
def bench_that_plus_di():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            that_ret = that_PLUS_DI(high, low, close, timeperiod=t)

if __name__ == '__main__':
    bench_this_plus_di()
    bench_that_plus_di() 
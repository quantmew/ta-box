from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import AROON as this_AROON
from talib import AROON as that_AROON

@bench
def bench_this_aroon():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        this_ret = this_AROON(high, low, timeperiod=t)

@bench
def bench_that_aroon():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        that_ret = that_AROON(high, low, timeperiod=t)

if __name__ == '__main__':
    bench_this_aroon()
    bench_that_aroon()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tabox import MAXINDEX as this_MAXINDEX
from talib import MAXINDEX as that_MAXINDEX
from utils import bench

@bench
def bench_this_maxindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        this_MAXINDEX(data, timeperiod=30)


@bench
def bench_that_maxindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        that_MAXINDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_maxindex()
    bench_that_maxindex()
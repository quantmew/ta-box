import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import talib
from tabox import MINMAXINDEX
from utils import *


def bench_this_minmaxindex():
    for i in range(100, 10000, 100):
        data = np.random.random(i)
        MINMAXINDEX(data, timeperiod=30)


def bench_that_minmaxindex():
    for i in range(100, 10000, 100):
        data = np.random.random(i)
        talib.MINMAXINDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_minmaxindex()
    bench_that_minmaxindex() 
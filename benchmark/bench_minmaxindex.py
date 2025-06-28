import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tabox import MINMAXINDEX as this_MINMAXINDEX
from talib import MINMAXINDEX as that_MINMAXINDEX
from utils import *


def bench_this_minmaxindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        this_MINMAXINDEX(data, timeperiod=30)


def bench_that_minmaxindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        that_MINMAXINDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_minmaxindex()
    bench_that_minmaxindex() 
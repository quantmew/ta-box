import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tabox import MININDEX as this_MININDEX
from talib import MININDEX as that_MININDEX
from utils import *


def bench_this_minindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        this_MININDEX(data, timeperiod=30)


def bench_that_minindex():
    for i in range(100, 2000):
        data = np.random.random(i)
        that_MININDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_minindex()
    bench_that_minindex() 
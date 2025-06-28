import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tabox import MINMAX as this_MINMAX
from talib import MINMAX as that_MINMAX
from utils import *


def bench_this_minmax():
    for i in range(100, 2000):
        data = np.random.random(i)
        this_MINMAX(data, timeperiod=30)


def bench_that_minmax():
    for i in range(100, 2000):
        data = np.random.random(i)
        that_MINMAX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_minmax()
    bench_that_minmax() 
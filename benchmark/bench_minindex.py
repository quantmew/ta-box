import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import talib
from tabox import MININDEX
from utils import *
from tqdm import tqdm


def bench_this_minindex():
    for i in tqdm(range(100, 10000, 100)):
        data = np.random.random(i)
        MININDEX(data, timeperiod=30)


def bench_that_minindex():
    for i in tqdm(range(100, 10000, 100)):
        data = np.random.random(i)
        talib.MININDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_minindex()
    bench_that_minindex() 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import talib
from tabox import MAXINDEX
from utils import *
from tqdm import tqdm


def bench_this_maxindex():
    for i in tqdm(range(100, 10000, 100)):
        data = np.random.random(i)
        MAXINDEX(data, timeperiod=30)


def bench_that_maxindex():
    for i in tqdm(range(100, 10000, 100)):
        data = np.random.random(i)
        talib.MAXINDEX(data, timeperiod=30)


if __name__ == "__main__":
    bench_this_maxindex()
    bench_that_maxindex() 
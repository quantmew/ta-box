from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MINUS_DM as this_MINUS_DM
from talib import MINUS_DM as that_MINUS_DM

@bench
def bench_this_minus_dm():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        this_ret = this_MINUS_DM(high, low, timeperiod=t)

@bench
def bench_that_minus_dm():
    for i in range(100, 2000):
        t = 14
        high = np.random.random(i)
        low = np.random.random(i)
        that_ret = that_MINUS_DM(high, low, timeperiod=t)

if __name__ == '__main__':
    bench_this_minus_dm()
    bench_that_minus_dm() 
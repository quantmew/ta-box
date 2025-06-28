from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import CORREL as this_CORREL
from talib import CORREL as that_CORREL

@bench
def bench_this_correl():
    for i in range(100, 2000):
        t = 14
        x = np.random.random(i)
        y = np.random.random(i)
        this_ret = this_CORREL(x, y, timeperiod=t)

@bench
def bench_that_correl():
    for i in range(100, 2000):
        t = 14
        x = np.random.random(i)
        y = np.random.random(i)
        that_ret = that_CORREL(x, y, timeperiod=t)

if __name__ == '__main__':
    bench_this_correl()
    bench_that_correl() 
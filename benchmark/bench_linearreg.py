from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import LINEARREG as this_LINEARREG
from talib import LINEARREG as that_LINEARREG

@bench
def bench_this_linearreg():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_LINEARREG(close, timeperiod=t)

@bench
def bench_that_linearreg():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_LINEARREG(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_linearreg()
    bench_that_linearreg() 
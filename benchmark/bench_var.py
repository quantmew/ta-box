from utils import bench
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import VAR as this_VAR
from talib import VAR as that_VAR

@bench
def bench_this_var():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_VAR(close, timeperiod=t)

@bench
def bench_that_var():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_VAR(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_var()
    bench_that_var() 
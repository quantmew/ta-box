from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import STOCHRSI as this_STOCHRSI
from talib import STOCHRSI as that_STOCHRSI

@bench
def bench_this_stochrsi():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_STOCHRSI(close, timeperiod=t)

@bench
def bench_that_stochrsi():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_STOCHRSI(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_stochrsi()
    bench_that_stochrsi() 
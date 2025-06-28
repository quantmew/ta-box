from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import OBV as this_OBV
from talib import OBV as that_OBV

@bench
def bench_this_obv():
    for i in range(100, 2000):
        close = np.random.random(i)
        volume = np.random.random(i)
        this_ret = this_OBV(close, volume)

@bench
def bench_that_obv():
    for i in range(100, 2000):
        close = np.random.random(i)
        volume = np.random.random(i)
        that_ret = that_OBV(close, volume)

if __name__ == '__main__':
    bench_this_obv()
    bench_that_obv() 
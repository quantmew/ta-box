from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ROCR as this_ROCR
from talib import ROCR as that_ROCR

@bench
def bench_this_rocr():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_ROCR(close, timeperiod=t)

@bench
def bench_that_rocr():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_ROCR(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_rocr()
    bench_that_rocr() 
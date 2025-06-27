from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import CMO as this_CMO
from talib import CMO as that_CMO

@bench
def bench_this_cmo():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            close = np.random.random(i)
            this_ret = this_CMO(close, timeperiod=t)

@bench
def bench_that_cmo():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            close = np.random.random(i)
            that_ret = that_CMO(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_cmo()
    bench_that_cmo() 
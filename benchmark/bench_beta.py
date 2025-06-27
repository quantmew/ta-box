from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import BETA as this_BETA
from talib import BETA as that_BETA

@bench
def bench_this_beta():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            x = np.random.random(i)
            y = np.random.random(i)
            this_ret = this_BETA(x, y, timeperiod=t)

@bench
def bench_that_beta():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            x = np.random.random(i)
            y = np.random.random(i)
            that_ret = that_BETA(x, y, timeperiod=t)

if __name__ == '__main__':
    bench_this_beta()
    bench_that_beta() 
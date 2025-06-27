from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import STOCH as this_STOCH
from talib import STOCH as that_STOCH

@bench
def bench_this_stoch():
    for i in range(100, 1000):
        for fastk in [5, 10, 14]:
            for slowk in [3, 5, 10]:
                for slowd in [3, 5, 10]:
                    high = np.random.random(i)
                    low = np.random.random(i)
                    close = np.random.random(i)
                    this_ret = this_STOCH(high, low, close, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)

@bench
def bench_that_stoch():
    for i in range(100, 1000):
        for fastk in [5, 10, 14]:
            for slowk in [3, 5, 10]:
                for slowd in [3, 5, 10]:
                    high = np.random.random(i)
                    low = np.random.random(i)
                    close = np.random.random(i)
                    that_ret = that_STOCH(high, low, close, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)

if __name__ == '__main__':
    bench_this_stoch()
    bench_that_stoch() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SAREXT as this_SAREXT
from talib import SAREXT as that_SAREXT

@bench
def bench_this_sarext():
    for i in range(100, 2000):
        high = np.random.random(i)
        low = np.random.random(i)
        this_ret = this_SAREXT(high, low)

@bench
def bench_that_sarext():
    for i in range(100, 2000):
        high = np.random.random(i)
        low = np.random.random(i)
        that_ret = that_SAREXT(high, low)

if __name__ == '__main__':
    bench_this_sarext()
    bench_that_sarext() 
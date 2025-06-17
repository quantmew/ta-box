from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import LN as this_LN
from talib import LN as that_LN

@bench
def bench_this_ln():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_LN(close)

@bench
def bench_that_ln():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_LN(close)

if __name__ == '__main__':
    bench_this_ln()
    bench_that_ln() 
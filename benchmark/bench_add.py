from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ADD as this_ADD
from talib import ADD as that_ADD

@bench
def bench_this_acos():
    for i in range(100, 10000):
        close1 = np.random.random(i)
        close2 = np.random.random(i)
        this_ret = this_ADD(close1, close2)

@bench
def bench_that_acos():
    for i in range(100, 10000):
        close1 = np.random.random(i)
        close2 = np.random.random(i)
        that_ret = that_ADD(close1, close2)

if __name__ == '__main__':
    bench_this_acos()
    bench_that_acos()

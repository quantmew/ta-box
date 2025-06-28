from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SUB as this_SUB
from talib import SUB as that_SUB

@bench
def bench_this_sub():
    for i in range(100, 2000):
        x = np.random.random(i)
        y = np.random.random(i)
        this_ret = this_SUB(x, y)

@bench
def bench_that_sub():
    for i in range(100, 2000):
        x = np.random.random(i)
        y = np.random.random(i)
        that_ret = that_SUB(x, y)

if __name__ == '__main__':
    bench_this_sub()
    bench_that_sub() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import DIV as this_DIV
from talib import DIV as that_DIV

@bench
def bench_this_div():
    for i in range(100, 2000):
        x = np.random.random(i)
        y = np.random.random(i)
        this_ret = this_DIV(x, y)

@bench
def bench_that_div():
    for i in range(100, 2000):
        x = np.random.random(i)
        y = np.random.random(i)
        that_ret = that_DIV(x, y)

if __name__ == '__main__':
    bench_this_div()
    bench_that_div() 
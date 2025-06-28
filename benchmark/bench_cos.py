from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import COS as this_COS
from talib import COS as that_COS

@bench
def bench_this_cos():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_COS(close)

@bench
def bench_that_cos():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_COS(close)

if __name__ == '__main__':
    bench_this_cos()
    bench_that_cos() 
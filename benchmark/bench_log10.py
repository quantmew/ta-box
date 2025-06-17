from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import LOG10 as this_LOG10
from talib import LOG10 as that_LOG10

@bench
def bench_this_log10():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_LOG10(close)

@bench
def bench_that_log10():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_LOG10(close)

if __name__ == '__main__':
    bench_this_log10()
    bench_that_log10() 
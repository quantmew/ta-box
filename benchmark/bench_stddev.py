from utils import bench
import numpy as np
import sys

sys.path.append('..')
from tabox import STDDEV as this_STDDEV
from talib import STDDEV as that_STDDEV

@bench
def bench_this_stddev():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_STDDEV(close)

@bench
def bench_that_stddev():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_STDDEV(close)

if __name__ == '__main__':
    bench_this_stddev()
    bench_that_stddev()
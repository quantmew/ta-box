from utils import bench
import numpy as np
import sys

sys.path.append('..')
from tabox import BBANDS as this_BBANDS
from talib import BBANDS as that_BBANDS

@bench
def bench_this_bbands():
    for i in range(100, 2000):
        close = np.random.random(i) * 100
        this_BBANDS(close)

@bench
def bench_that_bbands():
    for i in range(100, 2000):
        close = np.random.random(i) * 100
        that_BBANDS(close)

if __name__ == '__main__':
    bench_this_bbands()
    bench_that_bbands()
from utils import bench
import numpy as np
import sys
sys.path.append('..')

from tabox import MAMA as this_MAMA
from talib import MAMA as that_MAMA

@bench
def bench_this_mama():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_MAMA(close)

@bench
def bench_that_mama():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_MAMA(close)

if __name__ == '__main__':
    bench_this_mama()
    bench_that_mama() 
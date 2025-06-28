from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MACDFIX as this_MACDFIX
from talib import MACDFIX as that_MACDFIX

@bench
def bench_this_macdfix():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_MACDFIX(close)

@bench
def bench_that_macdfix():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_MACDFIX(close)

if __name__ == '__main__':
    bench_this_macdfix()
    bench_that_macdfix() 
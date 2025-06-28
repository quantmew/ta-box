from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MACDEXT as this_MACDEXT
from talib import MACDEXT as that_MACDEXT

@bench
def bench_this_macdext():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_MACDEXT(close)

@bench
def bench_that_macdext():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_MACDEXT(close)

if __name__ == '__main__':
    bench_this_macdext()
    bench_that_macdext() 
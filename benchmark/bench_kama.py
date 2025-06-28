from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import KAMA as this_KAMA
from talib import KAMA as that_KAMA

@bench
def bench_this_kama():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_kama = this_KAMA(close)

@bench
def bench_that_kama():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_kama = that_KAMA(close)

if __name__ == '__main__':
    bench_this_kama()
    bench_that_kama() 
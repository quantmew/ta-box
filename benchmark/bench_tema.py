from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TEMA as this_TEMA
from talib import TEMA as that_TEMA

@bench
def bench_this_tema():
    for i in range(100, 1000):
        close = np.random.random(i)
        this_tema = this_TEMA(close)

@bench
def bench_that_tema():
    for i in range(100, 1000):
        close = np.random.random(i)
        that_tema = that_TEMA(close)

if __name__ == '__main__':
    bench_this_tema()
    bench_that_tema() 
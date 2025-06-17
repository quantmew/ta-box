from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TRIMA as this_TRIMA
from talib import TRIMA as that_TRIMA

@bench
def bench_this_trima():
    for i in range(100, 1000):
        close = np.random.random(i)
        this_trima = this_TRIMA(close)

@bench
def bench_that_trima():
    for i in range(100, 1000):
        close = np.random.random(i)
        that_trima = that_TRIMA(close)

if __name__ == '__main__':
    bench_this_trima()
    bench_that_trima() 
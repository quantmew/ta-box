from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import TANH as this_TANH
from talib import TANH as that_TANH

@bench
def bench_this_tanh():
    for i in range(100, 10000):
        close = np.random.random(i)
        this_ret = this_TANH(close)

@bench
def bench_that_tanh():
    for i in range(100, 10000):
        close = np.random.random(i)
        that_ret = that_TANH(close)

if __name__ == '__main__':
    bench_this_tanh()
    bench_that_tanh() 
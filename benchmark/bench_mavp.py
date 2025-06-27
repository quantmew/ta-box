from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MAVP as this_MAVP
from talib import MAVP as that_MAVP

@bench
def bench_this_mavp():
    for i in range(100, 5000):
        real = np.random.random(i)
        periods = np.random.randint(2, 30, i).astype(np.float64)
        this_ret = this_MAVP(real, periods)

@bench
def bench_that_mavp():
    for i in range(100, 5000):
        real = np.random.random(i)
        periods = np.random.randint(2, 30, i).astype(np.float64)
        that_ret = that_MAVP(real, periods)

if __name__ == '__main__':
    bench_this_mavp()
    bench_that_mavp() 
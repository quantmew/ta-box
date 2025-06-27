from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import SAR as this_SAR
from talib import SAR as that_SAR

@bench
def bench_this_sar():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        this_ret = this_SAR(high, low)

@bench
def bench_that_sar():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        that_ret = that_SAR(high, low)

if __name__ == '__main__':
    bench_this_sar()
    bench_that_sar() 
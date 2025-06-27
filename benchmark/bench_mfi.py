from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MFI as this_MFI
from talib import MFI as that_MFI

@bench
def bench_this_mfi():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)
            this_ret = this_MFI(high, low, close, volume, timeperiod=t)

@bench
def bench_that_mfi():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)
            that_ret = that_MFI(high, low, close, volume, timeperiod=t)

if __name__ == '__main__':
    bench_this_mfi()
    bench_that_mfi() 
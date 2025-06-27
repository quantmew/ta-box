from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import AROONOSC as this_AROONOSC
from talib import AROONOSC as that_AROONOSC

@bench
def bench_this_aroonosc():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            this_ret = this_AROONOSC(high, low, timeperiod=t)

@bench
def bench_that_aroonosc():
    for i in range(100, 5000):
        for t in [7, 14, 21]:
            high = np.random.random(i)
            low = np.random.random(i)
            that_ret = that_AROONOSC(high, low, timeperiod=t)

if __name__ == '__main__':
    bench_this_aroonosc()
    bench_that_aroonosc() 
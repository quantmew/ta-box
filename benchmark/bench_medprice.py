from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MEDPRICE as this_MEDPRICE
from talib import MEDPRICE as that_MEDPRICE

@bench
def bench_this_medprice():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        this_ret = this_MEDPRICE(high, low)

@bench
def bench_that_medprice():
    for i in range(100, 5000):
        high = np.random.random(i)
        low = np.random.random(i)
        that_ret = that_MEDPRICE(high, low)

if __name__ == '__main__':
    bench_this_medprice()
    bench_that_medprice() 
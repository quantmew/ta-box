from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import MOM as this_MOM
from talib import MOM as that_MOM

@bench
def bench_this_mom():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_MOM(close, timeperiod=t)

@bench
def bench_that_mom():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_MOM(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_mom()
    bench_that_mom() 
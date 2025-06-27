from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ROCR100 as this_ROCR100
from talib import ROCR100 as that_ROCR100

@bench
def bench_this_rocr100():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_ROCR100(close, timeperiod=t)

@bench
def bench_that_rocr100():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_ROCR100(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_rocr100()
    bench_that_rocr100() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ROC as this_ROC
from talib import ROC as that_ROC

@bench
def bench_this_roc():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            this_ret = this_ROC(close, timeperiod=t)

@bench
def bench_that_roc():
    for i in range(100, 5000):
        for t in [5, 10, 20]:
            close = np.random.random(i)
            that_ret = that_ROC(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_roc()
    bench_that_roc() 
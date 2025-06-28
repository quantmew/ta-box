from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import APO as this_APO
from talib import APO as that_APO

@bench
def bench_this_apo():
    for i in range(100, 2000):
        fast = 3
        slow = 10
        close = np.random.random(i)
        this_ret = this_APO(close, fastperiod=fast, slowperiod=slow)

@bench
def bench_that_apo():
    for i in range(100, 2000):
        fast = 3
        slow = 10
        close = np.random.random(i)
        that_ret = that_APO(close, fastperiod=fast, slowperiod=slow)

if __name__ == '__main__':
    bench_this_apo()
    bench_that_apo() 
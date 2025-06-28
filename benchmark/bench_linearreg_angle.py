from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import LINEARREG_ANGLE as this_LINEARREG_ANGLE
from talib import LINEARREG_ANGLE as that_LINEARREG_ANGLE

@bench
def bench_this_linearreg_angle():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        this_ret = this_LINEARREG_ANGLE(close, timeperiod=t)

@bench
def bench_that_linearreg_angle():
    for i in range(100, 2000):
        t = 14
        close = np.random.random(i)
        that_ret = that_LINEARREG_ANGLE(close, timeperiod=t)

if __name__ == '__main__':
    bench_this_linearreg_angle()
    bench_that_linearreg_angle() 
from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import ADOSC as this_ADOSC
from talib import ADOSC as that_ADOSC

@bench
def bench_this_adosc():
    for i in range(100, 2000):
        for fast in [3, 5, 10]:
            for slow in [10, 20, 30]:
                high = np.random.random(i)
                low = np.random.random(i)
                close = np.random.random(i)
                volume = np.random.random(i)
                this_ret = this_ADOSC(high, low, close, volume, fast_period=fast, slow_period=slow)

@bench
def bench_that_adosc():
    for i in range(100, 2000):
        for fast in [3, 5, 10]:
            for slow in [10, 20, 30]:
                high = np.random.random(i)
                low = np.random.random(i)
                close = np.random.random(i)
                volume = np.random.random(i)
                that_ret = that_ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)

if __name__ == '__main__':
    bench_this_adosc()
    bench_that_adosc() 
from utils import bench
import numpy as np
import sys

sys.path.append('..')
from tabox import BBANDS as this_BBANDS
from talib import BBANDS as that_BBANDS

@bench
def bench_this_bbands():
    for i in range(1000, 2000, 100):
        close = np.random.random(i) * 100
        this_BBANDS(close)
        for timeperiod in [5, 20, 50]:
            for nbdev in [1.0, 2.0, 3.0]:
                this_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)

@bench
def bench_that_bbands():
    for i in range(1000, 2000, 100):
        close = np.random.random(i) * 100
        that_BBANDS(close)
        for timeperiod in [5, 20, 50]:
            for nbdev in [1.0, 2.0, 3.0]:
                that_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)

if __name__ == '__main__':
    bench_this_bbands()
    bench_that_bbands()
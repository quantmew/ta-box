from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import HT_TRENDLINE as this_HT_TRENDLINE
from talib import HT_TRENDLINE as that_HT_TRENDLINE

@bench
def bench_this_ht_trendline():
    for i in range(100, 2000):
        close = np.random.random(i)
        this_ret = this_HT_TRENDLINE(close)

@bench
def bench_that_ht_trendline():
    for i in range(100, 2000):
        close = np.random.random(i)
        that_ret = that_HT_TRENDLINE(close)

if __name__ == '__main__':
    bench_this_ht_trendline()
    bench_that_ht_trendline()
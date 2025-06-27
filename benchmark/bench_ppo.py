from utils import bench
import tqdm
import numpy as np

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from tabox import PPO as this_PPO
from talib import PPO as that_PPO

@bench
def bench_this_ppo():
    for i in range(100, 5000):
        for fast in [3, 5, 10]:
            for slow in [10, 20, 30]:
                close = np.random.random(i)
                this_ret = this_PPO(close, fastperiod=fast, slowperiod=slow)

@bench
def bench_that_ppo():
    for i in range(100, 5000):
        for fast in [3, 5, 10]:
            for slow in [10, 20, 30]:
                close = np.random.random(i)
                that_ret = that_PPO(close, fastperiod=fast, slowperiod=slow)

if __name__ == '__main__':
    bench_this_ppo()
    bench_that_ppo() 
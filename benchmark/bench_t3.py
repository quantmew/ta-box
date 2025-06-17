from utils import bench
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')

from tabox import T3 as this_T3
from talib import T3 as that_T3

@bench
def bench_this_t3():
    for i in tqdm(range(100, 1000)):
        close = np.random.random(i)
        this_T3(close)

@bench
def bench_that_t3():
    for i in tqdm(range(100, 1000)):
        close = np.random.random(i)
        that_T3(close)

if __name__ == '__main__':
    bench_this_t3()
    bench_that_t3() 
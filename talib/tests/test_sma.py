
from ..ta_func.ta_SMA import SMA as this_SMA
from talib import SMA as that_SMA
import time
import numpy as np
import tqdm


def test_sma():
    for i in tqdm.tqdm(range(100, 1000)):
        for t in [1, 2, 3, 5, 10, 30]:
            close = np.random.random(i)
            this_ret = this_SMA(close, timeperiod=t)
            that_ret = that_SMA(close, timeperiod=t)

            if not np.array_equal(this_ret, that_ret, equal_nan=True):
                print(i, t, this_ret, that_ret)
                break
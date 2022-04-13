
import numpy as np

from tabox.ta_func.ta_SMA import SMA as this_SMA
from talib import SMA as that_SMA

import unittest

class TestSMA(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 1000):
            t = 3
            close = np.random.random(i)
            this_ret = this_SMA(close, timeperiod=t)
            that_ret = that_SMA(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
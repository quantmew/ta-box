
import numpy as np

from tabox.ta_func.ta_RSI import RSI as this_RSI
from talib import RSI as that_RSI

import unittest

class TestRSI(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 200):
            t = 3
            close = np.random.random(i)
            this_ret = this_RSI(close, timeperiod=t)
            that_ret = that_RSI(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))
        
        for i in range(200, 300):
            t = 200
            close = np.random.random(i)
            this_ret = this_RSI(close, timeperiod=t)
            that_ret = that_RSI(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
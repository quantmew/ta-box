
import numpy as np

from tabox.ta_func.ta_MIN import MIN as this_MIN
from talib import MIN as that_MIN

import unittest

class TestMIN(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 1000):
            t = 3
            close = np.random.random(i)
            this_ret = this_MIN(close, timeperiod=t)
            that_ret = that_MIN(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))
        
        for i in range(200, 500):
            t = 200
            close = np.random.random(i)
            this_ret = this_MIN(close, timeperiod=t)
            that_ret = that_MIN(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
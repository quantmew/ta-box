
import numpy as np

from tabox.ta_func.ta_MAX import MAX as this_MAX
from talib import MAX as that_MAX

import unittest

class TestMAX(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 1000):
            t = 3
            close = np.random.random(i)
            this_ret = this_MAX(close, timeperiod=t)
            that_ret = that_MAX(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
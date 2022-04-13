
import numpy as np

from tabox import TAN as this_TAN
from talib import TAN as that_TAN

import unittest

class TestTAN(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_TAN(close)
            that_ret = that_TAN(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()

import numpy as np

from tabox import TRANGE as this_TRANGE
from talib import TRANGE as that_TRANGE

import unittest

class TestTRANGE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_TRANGE(close, high, low)
            that_ret = that_TRANGE(close, high, low)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
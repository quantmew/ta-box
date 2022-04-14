
import numpy as np

from tabox import ASIN as this_ASIN
from talib import ASIN as that_ASIN

import unittest

class TestASIN(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_ASIN(close)
            that_ret = that_ASIN(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main()
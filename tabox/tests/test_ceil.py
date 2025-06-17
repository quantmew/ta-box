import numpy as np

from tabox import CEIL as this_CEIL
from talib import CEIL as that_CEIL

import unittest

class TestCEIL(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_CEIL(close)
            that_ret = that_CEIL(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
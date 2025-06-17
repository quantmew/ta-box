import numpy as np

from tabox import SINH as this_SINH
from talib import SINH as that_SINH

import unittest

class TestSINH(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_SINH(close)
            that_ret = that_SINH(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
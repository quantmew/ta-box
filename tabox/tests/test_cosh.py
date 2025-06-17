import numpy as np

from tabox import COSH as this_COSH
from talib import COSH as that_COSH

import unittest

class TestCOSH(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_COSH(close)
            that_ret = that_COSH(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
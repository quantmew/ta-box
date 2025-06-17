import numpy as np

from tabox import LN as this_LN
from talib import LN as that_LN

import unittest

class TestLN(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_LN(close)
            that_ret = that_LN(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
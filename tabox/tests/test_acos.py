
import numpy as np

from tabox import ACOS as this_ACOS
from talib import ACOS as that_ACOS

import unittest

class TestACOS(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_ACOS(close)
            that_ret = that_ACOS(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main()
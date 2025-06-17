import numpy as np

from tabox import COS as this_COS
from talib import COS as that_COS

import unittest

class TestCOS(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_COS(close)
            that_ret = that_COS(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
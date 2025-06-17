import numpy as np

from tabox import FLOOR as this_FLOOR
from talib import FLOOR as that_FLOOR

import unittest

class TestFLOOR(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_FLOOR(close)
            that_ret = that_FLOOR(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
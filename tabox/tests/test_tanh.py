import numpy as np

from tabox import TANH as this_TANH
from talib import TANH as that_TANH

import unittest

class TestTANH(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_TANH(close)
            that_ret = that_TANH(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
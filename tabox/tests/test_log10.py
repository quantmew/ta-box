import numpy as np

from tabox import LOG10 as this_LOG10
from talib import LOG10 as that_LOG10

import unittest

class TestLOG10(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_LOG10(close)
            that_ret = that_LOG10(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
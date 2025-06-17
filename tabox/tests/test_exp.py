import numpy as np

from tabox import EXP as this_EXP
from talib import EXP as that_EXP

import unittest

class TestEXP(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_EXP(close)
            that_ret = that_EXP(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 

import numpy as np

from tabox import MULT as this_MULT
from talib import MULT as that_MULT

import unittest

class TestMULT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close1 = np.random.random(i)
            close2 = np.random.random(i)
            this_ret = this_MULT(close1, close2)
            that_ret = that_MULT(close1, close2)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close1}, {close2}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main()

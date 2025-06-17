
import numpy as np

from tabox import SUB as this_SUB
from talib import SUB as that_SUB

import unittest

class TestSUB(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close1 = np.random.random(i)
            close2 = np.random.random(i)
            this_ret = this_SUB(close1, close2)
            that_ret = that_SUB(close1, close2)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close1}, {close2}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main()

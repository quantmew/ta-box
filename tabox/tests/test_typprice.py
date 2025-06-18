import numpy as np
import unittest

from tabox import TYPPRICE as this_TYPPRICE
from talib import TYPPRICE as that_TYPPRICE

class TestTypPrice(unittest.TestCase):
    def test_typprice(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_TYPPRICE(high, low, close)
            that_ret = that_TYPPRICE(high, low, close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
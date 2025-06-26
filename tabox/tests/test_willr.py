import numpy as np
import unittest

from tabox import WILLR as this_WILLR
from talib import WILLR as that_WILLR

class TestWILLR(unittest.TestCase):
    def test_willr(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_WILLR(high, low, close)
            that_ret = that_WILLR(high, low, close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
import numpy as np
import unittest

from tabox import WCLPRICE as this_WCLPRICE
from talib import WCLPRICE as that_WCLPRICE

class TestWCLPrice(unittest.TestCase):
    def test_wclprice(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_WCLPRICE(high, low, close)
            that_ret = that_WCLPRICE(high, low, close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
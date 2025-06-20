import numpy as np
import unittest

from tabox import BOP as this_BOP
from talib import BOP as that_BOP

class TestBOP(unittest.TestCase):
    def test_BOP(self):
        for i in range(100, 300):
            open_ = np.random.random(i)
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_BOP(open_, high, low, close)
            that_ret = that_BOP(open_, high, low, close)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
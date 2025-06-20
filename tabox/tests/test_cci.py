import numpy as np
import unittest

from tabox import CCI as this_CCI
from talib import CCI as that_CCI

class TestCCI(unittest.TestCase):
    def test_CCI(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_CCI(high, low, close)
            that_ret = that_CCI(high, low, close)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
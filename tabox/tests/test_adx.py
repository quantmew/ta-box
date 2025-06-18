import numpy as np
import unittest

from tabox import ADX as this_ADX
from talib import ADX as that_ADX

class TestADX(unittest.TestCase):
    def test_adx(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_ADX(high, low, close)
            that_ret = that_ADX(high, low, close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
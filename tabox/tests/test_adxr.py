import numpy as np
import unittest

from tabox import ADXR as this_ADXR
from talib import ADXR as that_ADXR

class TestADXR(unittest.TestCase):
    def test_adxr(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_ADXR(high, low, close)
            that_ret = that_ADXR(high, low, close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
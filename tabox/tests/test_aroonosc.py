import numpy as np
import unittest

from tabox import AROONOSC as this_AROONOSC
from talib import AROONOSC as that_AROONOSC

class TestAROONOSC(unittest.TestCase):
    def test_AROONOSC(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            this_ret = this_AROONOSC(high, low)
            that_ret = that_AROONOSC(high, low)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
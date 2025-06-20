import numpy as np
import unittest

from tabox import AROON as this_AROON
from talib import AROON as that_AROON

class TestAROON(unittest.TestCase):
    def test_AROON(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            this_ret = this_AROON(high, low)
            that_ret = that_AROON(high, low)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
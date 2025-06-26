import numpy as np
import unittest

from tabox import ULTOSC as this_ULTOSC
from talib import ULTOSC as that_ULTOSC

class TestULTOSC(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 200):
            for timeperiod1, timeperiod2, timeperiod3 in [
                (3, 5, 10),
                (5, 10, 20),
                (10, 20, 30),
            ]:
                high = np.random.random(i)
                low = np.random.random(i)
                close = np.random.random(i)
                this_ret = this_ULTOSC(high, low, close, timeperiod1, timeperiod2, timeperiod3)
                that_ret = that_ULTOSC(high, low, close, timeperiod1, timeperiod2, timeperiod3)
                self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
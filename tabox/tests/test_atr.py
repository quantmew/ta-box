import numpy as np

from tabox import ATR as this_ATR
from talib import ATR as that_ATR

import unittest

class TestATR(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            
            # 确保high >= low
            high = np.maximum(high, low)
            
            this_ret = this_ATR(high, low, close)
            that_ret = that_ATR(high, low, close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
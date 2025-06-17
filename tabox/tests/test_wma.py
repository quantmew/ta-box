
import numpy as np

from tabox import WMA as this_WMA
from talib import WMA as that_WMA

import unittest

class TestTAN(unittest.TestCase):
    def test_random_vector(self):
        for k in [2,3,5,13,21]:
            for i in range(100, 1000):
                close = np.random.random(i)
                this_ret = this_WMA(close, k)
                that_ret = that_WMA(close, k)

                self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
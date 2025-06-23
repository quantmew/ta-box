
import numpy as np

from tabox import TSF as this_TSF
from talib import TSF as that_TSF

import unittest

class TestTSF(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_TSF(close)
            that_ret = that_TSF(close)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
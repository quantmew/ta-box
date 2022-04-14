
import numpy as np

from tabox import ATAN as this_ATAN
from talib import ATAN as that_ATAN

import unittest

class TestATAN(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_ATAN(close)
            that_ret = that_ATAN(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
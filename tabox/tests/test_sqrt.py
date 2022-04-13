
import numpy as np

from tabox import SQRT as this_SQRT
from talib import SQRT as that_SQRT

import unittest

class TestSQRT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_SQRT(close)
            that_ret = that_SQRT(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
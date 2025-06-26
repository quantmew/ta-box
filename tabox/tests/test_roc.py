
import numpy as np

from tabox import ROC as this_ROC
from talib import ROC as that_ROC

import unittest

class TestROC(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 200):
            t = 3
            close = np.random.random(i)
            this_ret = this_ROC(close, timeperiod=t)
            that_ret = that_ROC(close, timeperiod=t)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
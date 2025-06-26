
import numpy as np

from tabox import ROCR100 as this_ROCR100
from talib import ROCR100 as that_ROCR100

import unittest

class TestROCR100(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 200):
            for t in [3, 5, 10, 20, 30, 50]:
                close = np.random.random(i)
                this_ret = this_ROCR100(close, timeperiod=t)
                that_ret = that_ROCR100(close, timeperiod=t)

                self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
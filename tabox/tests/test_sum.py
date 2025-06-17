
import numpy as np

from tabox.ta_func.ta_SUM import SUM as this_SUM
from talib import SUM as that_SUM

import unittest

class TestSUM(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 300):
            for t in [3, 5, 7, 13, 30]:
                close = np.random.random(i)
                this_ret = this_SUM(close, timeperiod=t)
                that_ret = that_SUM(close, timeperiod=t)

                self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()


import numpy as np

from tabox import STOCH as this_STOCH
from talib import STOCH as that_STOCH

import unittest

class TestSTOCH(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            that_ret = that_STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
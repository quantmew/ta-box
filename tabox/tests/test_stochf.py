import numpy as np

from tabox import STOCHF as this_STOCHF
from talib import STOCHF as that_STOCHF

import unittest


class TestSTOCHF(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_STOCHF(
                high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
            )
            that_ret = that_STOCHF(
                high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
            )

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

import numpy as np

from tabox import STOCHRSI as this_STOCHRSI
from talib import STOCHRSI as that_STOCHRSI

import unittest


class TestSTOCHRSI(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_ret = this_STOCHRSI(
                close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
            )
            that_ret = that_STOCHRSI(
                close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
            )

            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

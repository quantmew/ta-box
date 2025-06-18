import numpy as np

from tabox import KAMA as this_KAMA
from talib import KAMA as that_KAMA

import unittest

class TestKAMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_kama = this_KAMA(close)
            that_kama = that_KAMA(close)

            self.assertTrue(np.allclose(this_kama, that_kama, equal_nan=True), f"{close}, {this_kama}, {that_kama}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_kama = this_KAMA(close, timeperiod=5)
        that_kama = that_KAMA(close, timeperiod=5)

        self.assertTrue(np.allclose(this_kama, that_kama, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
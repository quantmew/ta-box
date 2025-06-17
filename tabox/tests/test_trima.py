import numpy as np

from tabox import TRIMA as this_TRIMA
from talib import TRIMA as that_TRIMA

import unittest

class TestTRIMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_trima = this_TRIMA(close)
            that_trima = that_TRIMA(close)

            self.assertTrue(np.array_equal(this_trima, that_trima, equal_nan=True), f"{close}, {this_trima}, {that_trima}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_trima = this_TRIMA(close, timeperiod=5)
        that_trima = that_TRIMA(close, timeperiod=5)

        self.assertTrue(np.array_equal(this_trima, that_trima, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
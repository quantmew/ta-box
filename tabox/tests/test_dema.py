import numpy as np

from tabox import DEMA as this_DEMA
from talib import DEMA as that_DEMA

import unittest

class TestDEMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_dema = this_DEMA(close)
            that_dema = that_DEMA(close)

            self.assertTrue(np.array_equal(this_dema, that_dema, equal_nan=True), 
                          f"{close}, {this_dema}, {that_dema}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_dema = this_DEMA(close, timeperiod=5)
        that_dema = that_DEMA(close, timeperiod=5)

        self.assertTrue(np.array_equal(this_dema, that_dema, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
import numpy as np

from tabox import STDDEV as this_STDDEV
from talib import STDDEV as that_STDDEV

import unittest

class TestSTDDEV(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_stddev = this_STDDEV(close)
            that_stddev = that_STDDEV(close)

            self.assertTrue(np.array_equal(this_stddev, that_stddev, equal_nan=True), f"{close}, {this_stddev}, {that_stddev}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_stddev = this_STDDEV(close, timeperiod=10, nbdev=2.0)
        that_stddev = that_STDDEV(close, timeperiod=10, nbdev=2.0)

        self.assertTrue(np.array_equal(this_stddev, that_stddev, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
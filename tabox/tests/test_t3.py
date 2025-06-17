import numpy as np

from tabox import T3 as this_T3
from talib import T3 as that_T3

import unittest

class TestT3(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_t3 = this_T3(close)
            that_t3 = that_T3(close)

            self.assertTrue(np.array_equal(this_t3, that_t3, equal_nan=True), f"{close}, {this_t3}, {that_t3}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_t3 = this_T3(close, timeperiod=10, vfactor=0.5)
        that_t3 = that_T3(close, timeperiod=10, vfactor=0.5)

        self.assertTrue(np.array_equal(this_t3, that_t3, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
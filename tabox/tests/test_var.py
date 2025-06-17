import numpy as np

from tabox import VAR as this_VAR
from talib import VAR as that_VAR

import unittest

class TestVAR(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_var = this_VAR(close)
            that_var = that_VAR(close)

            self.assertTrue(np.array_equal(this_var, that_var, equal_nan=True), f"{close}, {this_var}, {that_var}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_var = this_VAR(close, timeperiod=10, nbdev=2.0)
        that_var = that_VAR(close, timeperiod=10, nbdev=2.0)

        self.assertTrue(np.array_equal(this_var, that_var, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
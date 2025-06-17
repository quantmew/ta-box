import numpy as np

from tabox import TEMA as this_TEMA
from talib import TEMA as that_TEMA

import unittest

class TestTEMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_tema = this_TEMA(close)
            that_tema = that_TEMA(close)

            self.assertTrue(np.array_equal(this_tema, that_tema, equal_nan=True), 
                          f"{close}, {this_tema}, {that_tema}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_tema = this_TEMA(close, timeperiod=5)
        that_tema = that_TEMA(close, timeperiod=5)

        self.assertTrue(np.array_equal(this_tema, that_tema, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
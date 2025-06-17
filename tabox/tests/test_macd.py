import numpy as np

from tabox import MACD as this_MACD
from talib import MACD as that_MACD

import unittest

class TestMACD(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_macd, this_signal, this_hist = this_MACD(close)
            that_macd, that_signal, that_hist = that_MACD(close)

            self.assertTrue(np.array_equal(this_macd, that_macd, equal_nan=True), f"{close}, {this_macd}, {that_macd}")
            self.assertTrue(np.array_equal(this_signal, that_signal, equal_nan=True), f"{close}, {this_signal}, {that_signal}")
            self.assertTrue(np.array_equal(this_hist, that_hist, equal_nan=True), f"{close}, {this_hist}, {that_hist}")

    def test_custom_periods(self):
        close = np.random.random(1000)
        this_macd, this_signal, this_hist = this_MACD(close, fastperiod=5, slowperiod=35, signalperiod=5)
        that_macd, that_signal, that_hist = that_MACD(close, fastperiod=5, slowperiod=35, signalperiod=5)

        self.assertTrue(np.array_equal(this_macd, that_macd, equal_nan=True))
        self.assertTrue(np.array_equal(this_signal, that_signal, equal_nan=True))
        self.assertTrue(np.array_equal(this_hist, that_hist, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
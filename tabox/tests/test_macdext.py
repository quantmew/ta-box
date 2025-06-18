import numpy as np

from tabox import MACDEXT as this_MACDEXT
from talib import MACDEXT as that_MACDEXT

import unittest

class TestMACDEXT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)

            this_macd, this_signal, this_hist = this_MACDEXT(
                close,
                fastperiod=12,
                fastmatype=0,
                slowperiod=26,
                slowmatype=0,
                signalperiod=9,
                signalmatype=0
            )
            that_macd, that_signal, that_hist = that_MACDEXT(
                close,
                fastperiod=12,
                fastmatype=0,
                slowperiod=26,
                slowmatype=0,
                signalperiod=9,
                signalmatype=0
            )

            self.assertTrue(np.allclose(this_macd, that_macd, equal_nan=True), f"{close}, {this_macd}, {that_macd}")
            self.assertTrue(np.allclose(this_signal, that_signal, equal_nan=True), f"{close}, {this_signal}, {that_signal}")
            self.assertTrue(np.allclose(this_hist, that_hist, equal_nan=True), f"{close}, {this_hist}, {that_hist}")

    def test_custom_periods_and_types(self):
        close = np.random.random(1000)
        this_macd, this_signal, this_hist = this_MACDEXT(
            close,
            fastperiod=5,
            fastmatype=1,
            slowperiod=35,
            slowmatype=2,
            signalperiod=5,
            signalmatype=3
        )
        that_macd, that_signal, that_hist = that_MACDEXT(
            close,
            fastperiod=5,
            fastmatype=1,
            slowperiod=35,
            slowmatype=2,
            signalperiod=5,
            signalmatype=3
        )

        self.assertTrue(np.allclose(this_macd, that_macd, equal_nan=True))
        self.assertTrue(np.allclose(this_signal, that_signal, equal_nan=True))
        self.assertTrue(np.allclose(this_hist, that_hist, equal_nan=True))

if __name__ == '__main__':
    unittest.main()

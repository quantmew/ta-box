import numpy as np

from tabox import MACDFIX as this_MACDFIX
from talib import MACDFIX as that_MACDFIX

import unittest


class TestMACDFIX(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)

            this_macd, this_signal, this_hist = this_MACDFIX(
                close,
                signalperiod=9,
            )
            that_macd, that_signal, that_hist = that_MACDFIX(
                close,
                signalperiod=9,
            )

            self.assertTrue(
                np.allclose(this_macd, that_macd, equal_nan=True),
                f"{close}, {this_macd}, {that_macd}",
            )
            self.assertTrue(
                np.allclose(this_signal, that_signal, equal_nan=True),
                f"{close}, {this_signal}, {that_signal}",
            )
            self.assertTrue(
                np.allclose(this_hist, that_hist, equal_nan=True),
                f"{close}, {this_hist}, {that_hist}",
            )

    def test_custom_periods_and_types(self):
        close = np.random.random(1000)
        this_macd, this_signal, this_hist = this_MACDFIX(
            close,
            signalperiod=5,
        )
        that_macd, that_signal, that_hist = that_MACDFIX(
            close,
            signalperiod=5,
        )

        self.assertTrue(np.allclose(this_macd, that_macd, equal_nan=True))
        self.assertTrue(np.allclose(this_signal, that_signal, equal_nan=True))
        self.assertTrue(np.allclose(this_hist, that_hist, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

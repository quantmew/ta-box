import numpy as np
import unittest

from tabox import MFI as this_MFI
from talib import MFI as that_MFI

class TestMFI(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)

            this_mfi = this_MFI(high, low, close, volume, timeperiod=14)
            that_mfi = that_MFI(high, low, close, volume, timeperiod=14)

            self.assertTrue(np.allclose(this_mfi, that_mfi, equal_nan=True), f"{high}, {low}, {close}, {volume}, {this_mfi}, {that_mfi}")

    def test_custom_period(self):
        high = np.random.random(1000)
        low = np.random.random(1000)
        close = np.random.random(1000)
        volume = np.random.random(1000)

        this_mfi = this_MFI(high, low, close, volume, timeperiod=5)
        that_mfi = that_MFI(high, low, close, volume, timeperiod=5)
    
        self.assertTrue(np.allclose(this_mfi, that_mfi, equal_nan=True), f"{high}, {low}, {close}, {volume}, {this_mfi}, {that_mfi}")

if __name__ == "__main__":
    unittest.main()
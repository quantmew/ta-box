import numpy as np
import unittest

from tabox import MIDPRICE as this_MIDPRICE
from talib import MIDPRICE as that_MIDPRICE

class TestMIDPRICE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)

            this_midprice = this_MIDPRICE(high, low, timeperiod=14)
            that_midprice = that_MIDPRICE(high, low, timeperiod=14)

            self.assertTrue(np.allclose(this_midprice, that_midprice, equal_nan=True), f"{high}, {low}, {this_midprice}, {that_midprice}")

if __name__ == "__main__":
    unittest.main()

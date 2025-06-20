import numpy as np
import unittest

from tabox import MIDPOINT as this_MIDPOINT
from talib import MIDPOINT as that_MIDPOINT

class TestMIDPOINT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            real = np.random.random(i)

            this_midpoint = this_MIDPOINT(real, timeperiod=14)
            that_midpoint = that_MIDPOINT(real, timeperiod=14)

            self.assertTrue(np.allclose(this_midpoint, that_midpoint, equal_nan=True), f"{real}, {this_midpoint}, {that_midpoint}")

if __name__ == "__main__":
    unittest.main()
import numpy as np
import unittest

from tabox import OBV as this_OBV
from talib import OBV as that_OBV

class TestOBV(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            real = np.random.random(i)
            volume = np.random.random(i)
            this_result = this_OBV(real, volume)
            that_result = that_OBV(real, volume)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
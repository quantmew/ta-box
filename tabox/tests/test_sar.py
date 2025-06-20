import numpy as np
import unittest

from tabox import SAR as this_SAR
from talib import SAR as that_SAR

class TestSAR(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            acceleration = np.random.random()
            maximum = np.random.random()
            this_result = this_SAR(high, low, acceleration, maximum)
            that_result = that_SAR(high, low, acceleration, maximum)

            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
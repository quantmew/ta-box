import numpy as np
import unittest

from tabox import MINUS_DM as this_MINUS_DM
from talib import MINUS_DM as that_MINUS_DM

class TestMINUS_DM(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            this_result = this_MINUS_DM(high, low)
            that_result = that_MINUS_DM(high, low)

            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
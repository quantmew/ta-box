import numpy as np
import unittest

from tabox import PLUS_DM as this_PLUS_DM
from talib import PLUS_DM as that_PLUS_DM

class TestPLUS_DM(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_result = this_PLUS_DM(high, low)
            that_result = that_PLUS_DM(high, low)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
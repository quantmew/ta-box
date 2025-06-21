import numpy as np
import unittest

from tabox import NATR as this_NATR
from talib import NATR as that_NATR

class TestNATR(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_result = this_NATR(high, low, close)
            that_result = that_NATR(high, low, close)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
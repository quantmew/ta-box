import numpy as np
import unittest

from tabox import MINUS_DI as this_MINUS_DI
from talib import MINUS_DI as that_MINUS_DI

class TestMINUS_DI(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_result = this_MINUS_DI(high, low, close)
            that_result = that_MINUS_DI(high, low, close)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
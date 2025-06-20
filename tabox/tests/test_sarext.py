import numpy as np
import unittest

from tabox import SAREXT as this_SAREXT
from talib import SAREXT as that_SAREXT


class TestSAREXT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            start = np.random.random(i)
            end = np.random.random(i)
            this_result = this_SAREXT(high, low)
            that_result = that_SAREXT(high, low)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

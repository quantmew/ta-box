import numpy as np
import unittest
from tabox import BETA as this_BETA
from talib import BETA as that_BETA

class TestBeta(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            real0 = np.random.random(i)
            real1 = np.random.random(i)
            this_result = this_BETA(real0, real1)
            that_result = that_BETA(real0, real1)

            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
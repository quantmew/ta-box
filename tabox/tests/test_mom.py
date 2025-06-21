import numpy as np
import unittest
from tabox import MOM as this_MOM
from talib import MOM as that_MOM

class TestMOM(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            real = np.random.random(i)
            this_result = this_MOM(real)
            that_result = that_MOM(real)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
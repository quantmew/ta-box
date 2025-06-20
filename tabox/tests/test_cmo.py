import numpy as np
import unittest

from tabox import CMO as this_CMO
from talib import CMO as that_CMO

class TestCMO(unittest.TestCase):
    def test_CMO(self):
        for i in range(100, 300):
            real = np.random.random(i)
            this_ret = this_CMO(real)
            that_ret = that_CMO(real)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
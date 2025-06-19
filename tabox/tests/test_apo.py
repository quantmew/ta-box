import numpy as np
import unittest

from tabox import APO as this_APO
from talib import APO as that_APO

class TestAPO(unittest.TestCase):
    def test_apo(self):
        for i in range(100, 300):
            real = np.random.random(i)
            this_ret = this_APO(real)
            that_ret = that_APO(real)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
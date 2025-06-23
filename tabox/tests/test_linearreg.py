import numpy as np

from tabox import LINEARREG as this_LINEARREG
from talib import LINEARREG as that_LINEARREG

import unittest

class TestLINEARREG(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_linearreg = this_LINEARREG(close)
            that_linearreg = that_LINEARREG(close)

            self.assertTrue(np.allclose(this_linearreg, that_linearreg, equal_nan=True), f"{close}, {this_linearreg}, {that_linearreg}")

if __name__ == '__main__':
    unittest.main() 
import numpy as np

from tabox import MAMA as this_MAMA
from talib import MAMA as that_MAMA

import unittest

class TestMAMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_mama, this_fama = this_MAMA(close)
            that_mama, that_fama = that_MAMA(close)

            self.assertTrue(np.allclose(this_mama, that_mama, equal_nan=True), f"{close}, {this_mama}, {that_mama}")
            self.assertTrue(np.allclose(this_fama, that_fama, equal_nan=True), f"{close}, {this_fama}, {that_fama}")

    def test_custom_limits(self):
        close = np.random.random(1000)
        this_mama, this_fama = this_MAMA(close, fastlimit=0.3, slowlimit=0.1)
        that_mama, that_fama = that_MAMA(close, fastlimit=0.3, slowlimit=0.1)

        self.assertTrue(np.allclose(this_mama, that_mama, equal_nan=True))
        self.assertTrue(np.allclose(this_fama, that_fama, equal_nan=True))

if __name__ == '__main__':
    unittest.main() 
import numpy as np

from tabox import LINEARREG_SLOPE as this_LINEARREG_SLOPE
from talib import LINEARREG_SLOPE as that_LINEARREG_SLOPE

import unittest


class TestLINEARREG_SLOPE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_linearreg_slope = this_LINEARREG_SLOPE(close)
            that_linearreg_slope = that_LINEARREG_SLOPE(close)

            self.assertTrue(
                np.allclose(this_linearreg_slope, that_linearreg_slope, equal_nan=True),
                f"{close}, {this_linearreg_slope}, {that_linearreg_slope}",
            )


if __name__ == "__main__":
    unittest.main()

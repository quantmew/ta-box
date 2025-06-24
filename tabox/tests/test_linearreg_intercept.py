import numpy as np

from tabox import LINEARREG_INTERCEPT as this_LINEARREG_INTERCEPT
from talib import LINEARREG_INTERCEPT as that_LINEARREG_INTERCEPT

import unittest


class TestLINEARREG_INTERCEPT(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_linearreg_intercept = this_LINEARREG_INTERCEPT(close)
            that_linearreg_intercept = that_LINEARREG_INTERCEPT(close)

            self.assertTrue(
                np.allclose(
                    this_linearreg_intercept, that_linearreg_intercept, equal_nan=True
                ),
                f"{close}, {this_linearreg_intercept}, {that_linearreg_intercept}",
            )


if __name__ == "__main__":
    unittest.main()

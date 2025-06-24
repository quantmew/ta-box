import numpy as np

from tabox import LINEARREG_ANGLE as this_LINEARREG_ANGLE
from talib import LINEARREG_ANGLE as that_LINEARREG_ANGLE

import unittest


class TestLINEARREG_ANGLE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            close = np.random.random(i)
            this_linearreg_angle = this_LINEARREG_ANGLE(close)
            that_linearreg_angle = that_LINEARREG_ANGLE(close)

            self.assertTrue(
                np.allclose(this_linearreg_angle, that_linearreg_angle, equal_nan=True),
                f"{close}, {this_linearreg_angle}, {that_linearreg_angle}",
            )


if __name__ == "__main__":
    unittest.main()

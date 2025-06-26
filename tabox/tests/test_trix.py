import numpy as np

from tabox import TRIX as this_TRIX
from talib import TRIX as that_TRIX

import unittest


class TestTRIX(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            for j in [3, 5, 10, 20, 30, 50]:
                close = np.random.random(i)
                this_trix = this_TRIX(close, timeperiod=j)
                that_trix = that_TRIX(close, timeperiod=j)

            self.assertTrue(
                np.allclose(this_trix, that_trix, equal_nan=True),
                f"{close}, {this_trix}, {that_trix}",
            )


if __name__ == "__main__":
    unittest.main()

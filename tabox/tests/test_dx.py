import numpy as np
import unittest

from tabox import DX as this_DX
from talib import DX as that_DX


class TestDX(unittest.TestCase):
    def test_dx(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)

            this_ret = this_DX(high, low, close)
            that_ret = that_DX(high, low, close)

            self.assertTrue(
                np.allclose(this_ret, that_ret, equal_nan=True),
                f"{high}, {low}, {close}, {this_ret}, {that_ret}",
            )


if __name__ == "__main__":
    unittest.main()

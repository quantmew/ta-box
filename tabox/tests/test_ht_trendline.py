import numpy as np

from tabox import HT_TRENDLINE as this_HT_TRENDLINE
from talib import HT_TRENDLINE as that_HT_TRENDLINE

import unittest

class TestHT_TRENDLINE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000):
            close = np.random.random(i)
            this_ret = this_HT_TRENDLINE(close)
            that_ret = that_HT_TRENDLINE(close)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), f"{close}, {this_ret}, {that_ret}")

if __name__ == '__main__':
    unittest.main() 
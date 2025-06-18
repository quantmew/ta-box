import numpy as np
import talib
from tabox import MAXINDEX as this_MAXINDEX
from talib import MAXINDEX as that_MAXINDEX

import unittest

class TestMAXINDEX(unittest.TestCase):
    def test_random_vector(self):
        np.random.seed(42)
        data = np.random.random(100)

        result = this_MAXINDEX(data, timeperiod=30)

        expected = that_MAXINDEX(data, timeperiod=30)

        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
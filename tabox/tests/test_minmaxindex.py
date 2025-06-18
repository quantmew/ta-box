import numpy as np
import talib
from tabox import MINMAXINDEX as this_MINMAXINDEX
from talib import MINMAXINDEX as that_MINMAXINDEX

import unittest

class TestMINMAXINDEX(unittest.TestCase):
    def test_random_vector(self):
        np.random.seed(42)
        data = np.random.random(100)

        minidx_result, maxidx_result = this_MINMAXINDEX(data, timeperiod=30)

        minidx_expected, maxidx_expected = that_MINMAXINDEX(data, timeperiod=30)

        np.testing.assert_array_equal(minidx_result, minidx_expected)
        np.testing.assert_array_equal(maxidx_result, maxidx_expected)

if __name__ == '__main__':
    unittest.main() 
import numpy as np
import talib
from tabox import MINMAX as this_MINMAX
from talib import MINMAX as that_MINMAX

import unittest

class TestMINMAX(unittest.TestCase):
    def test_random_vector(self):
        np.random.seed(42)
        data = np.random.random(100)
        
        min_result, max_result = this_MINMAX(data, timeperiod=30)

        min_expected, max_expected = that_MINMAX(data, timeperiod=30)

        np.testing.assert_array_equal(min_result, min_expected)
        np.testing.assert_array_equal(max_result, max_expected)

if __name__ == '__main__':
    unittest.main() 
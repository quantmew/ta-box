import numpy as np
import talib
from tabox import MININDEX as this_MININDEX
from talib import MININDEX as that_MININDEX


import unittest

class TestMININDEX(unittest.TestCase):
    def test_random_vector(self):
        np.random.seed(42)
        data = np.random.random(100)

        result = this_MININDEX(data, timeperiod=30)
        
        expected = that_MININDEX(data, timeperiod=30)

        np.testing.assert_array_equal(result, expected) 
if __name__ == '__main__':
    unittest.main() 
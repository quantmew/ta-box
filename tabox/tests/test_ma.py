import numpy as np

from tabox.ta_func.ta_MA import MA as this_MA
from talib import MA as that_MA

import unittest

class TestMA(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 300):
            for t in [3, 5, 7, 13, 30]:
                for matype in range(9):  # Test all 9 MA types
                    close = np.random.random(i)
                    this_ret = this_MA(close, timeperiod=t, matype=matype)
                    that_ret = that_MA(close, timeperiod=t, matype=matype)

                    self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))

    def test_ma_types(self):
        """Test all supported MA types"""
        close = np.random.random(100)
        
        # Test each MA type individually
        for matype in range(9):
            result = this_MA(close, timeperiod=14, matype=matype)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), len(close))

    def test_edge_cases(self):
        """Test edge cases"""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test minimum period
        result = this_MA(close, timeperiod=1, matype=0)
        np.testing.assert_array_equal(result, close)
        
        # Test period larger than data
        result = this_MA(close, timeperiod=10, matype=0)
        self.assertTrue(np.all(np.isnan(result)))

if __name__ == '__main__':
    unittest.main()
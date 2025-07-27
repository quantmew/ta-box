import unittest
import numpy as np
from tabox.ta_func.ta_utils import (
    check_array, check_timeperiod, check_begidx1, check_begidx2, 
    check_begidx3, check_begidx4, check_length2, check_length3, 
    check_length4, make_double_array
)

class TestTAUtils(unittest.TestCase):

    def test_check_array(self):
        """Test array validation function"""
        # Test numpy array input
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = check_array(arr)
        np.testing.assert_array_equal(result, arr)
        
        # Test list input
        lst = [1.0, 2.0, 3.0]
        result = check_array(lst)
        expected = np.array(lst, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)
        
        # Test wrong dtype
        arr_int = np.array([1, 2, 3], dtype=np.int32)
        with self.assertRaises(Exception):
            check_array(arr_int)
        
        # Test wrong dimensions
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(Exception):
            check_array(arr_2d)

    def test_check_timeperiod(self):
        """Test timeperiod validation"""
        # Valid timeperiod
        try:
            check_timeperiod(5)
        except Exception:
            self.fail("check_timeperiod raised exception for valid timeperiod")
        
        # Invalid timeperiods
        with self.assertRaises(Exception):
            check_timeperiod(1)
        with self.assertRaises(Exception):
            check_timeperiod(0)
        with self.assertRaises(Exception):
            check_timeperiod(-1)

    def test_check_begidx1(self):
        """Test single array begin index check"""
        # Normal array
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = check_begidx1(arr)
        self.assertEqual(result, 0)
        
        # Array with leading NaN
        arr_nan = np.array([np.nan, np.nan, 1.0, 2.0], dtype=np.float64)
        result = check_begidx1(arr_nan)
        self.assertEqual(result, 2)
        
        # All NaN array
        arr_all_nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        with self.assertRaises(Exception):
            check_begidx1(arr_all_nan)

    def test_check_begidx2(self):
        """Test dual array begin index check"""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = check_begidx2(arr1, arr2)
        self.assertEqual(result, 0)
        
        # With NaN in first array
        arr1_nan = np.array([np.nan, 2.0, 3.0], dtype=np.float64)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = check_begidx2(arr1_nan, arr2)
        self.assertEqual(result, 1)

    def test_check_length2(self):
        """Test dual array length validation"""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        length = check_length2(arr1, arr2)
        self.assertEqual(length, 3)
        
        # Different lengths
        arr1 = np.array([1.0, 2.0], dtype=np.float64)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        with self.assertRaises(Exception):
            check_length2(arr1, arr2)

    def test_make_double_array(self):
        """Test double array creation"""
        length = 5
        lookback = 2
        result = make_double_array(length, lookback)
        
        self.assertEqual(len(result), length)
        self.assertEqual(result.dtype, np.float64)
        
        # Check that first 'lookback' elements are NaN
        for i in range(lookback):
            self.assertTrue(np.isnan(result[i]))
        
        # Check that remaining elements are not NaN (they should be uninitialized but not NaN)
        # Actually, np.empty might contain arbitrary values, so we just check length and dtype

if __name__ == '__main__':
    unittest.main()
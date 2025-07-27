import unittest
import numpy as np
from tabox.ta_func.ta_defs import (
    TA_INTEGER_MIN, TA_INTEGER_MAX, TA_REAL_MIN, TA_REAL_MAX,
    TA_INTEGER_DEFAULT, TA_REAL_DEFAULT
)

class TestTADefs(unittest.TestCase):

    def test_constants_values(self):
        """Test that constants have expected values"""
        self.assertEqual(TA_INTEGER_MIN, -2147483647)
        self.assertEqual(TA_INTEGER_MAX, 2147483647)
        self.assertEqual(TA_INTEGER_DEFAULT, -2147483648)
        
        self.assertEqual(TA_REAL_MIN, -3e37)
        self.assertEqual(TA_REAL_MAX, 3e37)
        self.assertEqual(TA_REAL_DEFAULT, -4e37)

    def test_constants_relationships(self):
        """Test relationships between constants"""
        self.assertLess(TA_INTEGER_MIN, TA_INTEGER_MAX)
        self.assertLess(TA_INTEGER_DEFAULT, TA_INTEGER_MIN)
        
        self.assertLess(TA_REAL_MIN, TA_REAL_MAX)
        self.assertLess(TA_REAL_DEFAULT, TA_REAL_MIN)

    def test_default_values_outside_range(self):
        """Test that default values are outside valid ranges"""
        self.assertLess(TA_INTEGER_DEFAULT, TA_INTEGER_MIN)
        self.assertLess(TA_REAL_DEFAULT, TA_REAL_MIN)

if __name__ == '__main__':
    unittest.main()
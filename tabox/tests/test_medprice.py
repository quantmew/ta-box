import numpy as np
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from tabox import MEDPRICE as this_MEDPRICE
from talib import MEDPRICE as that_MEDPRICE

class TestMEDPRICE(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 1000, 100):
            open = np.random.random(i)
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            this_ret = this_MEDPRICE(high, low)
            that_ret = that_MEDPRICE(high, low)
            self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True, rtol=1e-5, atol=1e-8), 
                            f"MEDPRICE mismatch at length {i}: {high[:5]}...")

if __name__ == '__main__':
    unittest.main()
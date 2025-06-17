import numpy as np

from tabox import AD as this_AD
from talib import AD as that_AD

import unittest

class TestAD(unittest.TestCase):
    def test_random_vector(self):
        for i in range(10, 100):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)
            
            # Ensure high >= low
            high = np.maximum(high, low)
            
            this_ret = this_AD(high, low, close, volume)
            that_ret = that_AD(high, low, close, volume)

            self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), 
                          f"high: {high}, low: {low}, close: {close}, volume: {volume}, this_ret: {this_ret}, that_ret: {that_ret}")

if __name__ == '__main__':
    unittest.main()
import numpy as np

from tabox import EMA as this_EMA
from talib import EMA as that_EMA

import unittest

class TestEMA(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            for t in [3, 5, 7, 13, 30]:
                close = np.random.random(i)
                timeperiod = t
                
                this_ret = this_EMA(close, timeperiod)
                that_ret = that_EMA(close, timeperiod)

                self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), 
                            f"close: {close}, timeperiod: {timeperiod}, this_ret: {this_ret}, that_ret: {that_ret}")

if __name__ == '__main__':
    unittest.main()

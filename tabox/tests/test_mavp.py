import numpy as np
import unittest
from tabox import MAVP as this_MAVP
from talib import MAVP as that_MAVP

class TestMAVP(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            real = np.random.random(i)
            periods = np.random.random(i)
            minperiod = 2
            maxperiod = 5
            matype = 0
            this_result = this_MAVP(real, periods, minperiod, maxperiod, matype)
            that_result = that_MAVP(real, periods, minperiod, maxperiod, matype)
            self.assertTrue(np.allclose(this_result, that_result, equal_nan=True))

if __name__ == "__main__":
    unittest.main()
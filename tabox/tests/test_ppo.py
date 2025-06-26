import numpy as np

from tabox import PPO as this_PPO
from talib import PPO as that_PPO

import unittest


class TestPPO(unittest.TestCase):

    def test_random_vector(self):
        for i in range(100, 200):
            for fast_period, slow_period, matype in [
                (3, 5, 0),
                (5, 10, 1),
                (10, 20, 2),
                (20, 30, 3),
                (30, 50, 4),
            ]:
                close = np.random.random(i)
                this_ret = this_PPO(
                    close, fastperiod=fast_period, slowperiod=slow_period, matype=matype
                )
                that_ret = that_PPO(
                    close, fastperiod=fast_period, slowperiod=slow_period, matype=matype
                )

                self.assertTrue(np.allclose(this_ret, that_ret, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

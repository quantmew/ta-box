import numpy as np

from tabox import ADOSC as this_ADOSC
from talib import ADOSC as that_ADOSC

import unittest


class TestADOSC(unittest.TestCase):
    def test_random_vector(self):
        for i in range(100, 300):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)

            # 确保 high >= low
            high = np.maximum(high, low)

            # 测试不同的参数组合
            periods = [(3, 5), (5, 10), (7, 13), (13, 30)]

            for slow_period, fast_period in periods:
                this_ret = this_ADOSC(
                    high,
                    low,
                    close,
                    volume,
                    fast_period=fast_period,
                    slow_period=slow_period,
                )
                that_ret = that_ADOSC(
                    high,
                    low,
                    close,
                    volume,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                )

                self.assertTrue(
                    np.array_equal(this_ret, that_ret, equal_nan=True),
                    f"fast_period: {fast_period}, slow_period: {slow_period}, "
                    f"high: {high}, low: {low}, close: {close}, volume: {volume}, "
                    f"this_ret: {this_ret}, that_ret: {that_ret}",
                )


if __name__ == "__main__":
    unittest.main()

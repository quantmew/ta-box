import numpy as np

from tabox import ADOSC as this_ADOSC
from talib import ADOSC as that_ADOSC

import unittest

class TestADOSC(unittest.TestCase):
    def test_random_vector(self):
        for i in range(10, 100):
            high = np.random.random(i)
            low = np.random.random(i)
            close = np.random.random(i)
            volume = np.random.random(i)
            
            # 确保 high >= low
            high = np.maximum(high, low)
            
            # 测试不同的参数组合
            fast_periods = [3, 5, 10]
            slow_periods = [10, 20, 30]
            
            for fast_period in fast_periods:
                for slow_period in slow_periods:
                    this_ret = this_ADOSC(high, low, close, volume, 
                                        fast_period=fast_period, 
                                        slow_period=slow_period)
                    that_ret = that_ADOSC(high, low, close, volume, 
                                        fastperiod=fast_period, 
                                        slowperiod=slow_period)

                    self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True), 
                                  f"fast_period: {fast_period}, slow_period: {slow_period}, "
                                  f"high: {high}, low: {low}, close: {close}, volume: {volume}, "
                                  f"this_ret: {this_ret}, that_ret: {that_ret}")

    def test_edge_cases(self):
        # 测试相同价格
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0])
        volume = np.array([1000, 1000, 1000])
        
        this_ret = this_ADOSC(high, low, close, volume)
        that_ret = that_ADOSC(high, low, close, volume)
        
        self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))
        
        # 测试零交易量
        volume_zero = np.zeros_like(volume)
        this_ret = this_ADOSC(high, low, close, volume_zero)
        that_ret = that_ADOSC(high, low, close, volume_zero)
        
        self.assertTrue(np.array_equal(this_ret, that_ret, equal_nan=True))

    def test_invalid_input(self):
        # 测试无效的fast_period
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])
        volume = np.array([1000, 2000, 3000])
        
        with self.assertRaises(Exception):
            this_ADOSC(high, low, close, volume, fast_period=1)
        
        # 测试无效的slow_period
        with self.assertRaises(Exception):
            this_ADOSC(high, low, close, volume, slow_period=1)
        
        # 测试数组长度不匹配
        with self.assertRaises(Exception):
            this_ADOSC(high, low, close, volume[:-1])

if __name__ == '__main__':
    unittest.main()

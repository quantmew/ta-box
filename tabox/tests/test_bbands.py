import numpy as np
from tabox import BBANDS as this_BBANDS
from talib import BBANDS as that_BBANDS
import unittest

class TestBBANDS(unittest.TestCase):
    def test_random_vector(self):
        """测试随机数据向量上BBANDS函数的输出是否与TA-Lib一致"""
        # 测试不同长度的输入数组
        for i in range(100, 1000, 100):
            # 生成随机价格数据
            close = np.random.random(i) * 100  # 模拟价格在0-100之间
            
            # 测试默认参数
            this_upper, this_middle, this_lower = this_BBANDS(close)
            that_upper, that_middle, that_lower = that_BBANDS(close)
            
            # 验证输出形状一致
            self.assertEqual(this_upper.shape, that_upper.shape)
            self.assertEqual(this_middle.shape, that_middle.shape)
            self.assertEqual(this_lower.shape, that_lower.shape)
            
            # 验证计算结果在容差范围内一致
            self.assertTrue(np.allclose(this_upper, that_upper, equal_nan=True, rtol=1e-5, atol=1e-8), 
                           f"Upper band mismatch at length {i}: {close[:5]}...")
            self.assertTrue(np.allclose(this_middle, that_middle, equal_nan=True, rtol=1e-5, atol=1e-8), 
                           f"Middle band mismatch at length {i}: {close[:5]}...")
            self.assertTrue(np.allclose(this_lower, that_lower, equal_nan=True, rtol=1e-5, atol=1e-8), 
                           f"Lower band mismatch at length {i}: {close[:5]}...")
            
            # 测试不同参数组合
            for timeperiod in [5, 20, 50]:
                for nbdev in [1.0, 2.0, 3.0]:
                    # 使用相同的上轨和下轨系数
                    this_u, this_m, this_l = this_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
                    that_u, that_m, that_l = that_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
                    
                    self.assertTrue(np.allclose(this_u, that_u, equal_nan=True, rtol=1e-5, atol=1e-8), 
                                   f"Upper band mismatch at timeperiod={timeperiod}, nbdev={nbdev}")
                    self.assertTrue(np.allclose(this_m, that_m, equal_nan=True, rtol=1e-5, atol=1e-8), 
                                   f"Middle band mismatch at timeperiod={timeperiod}, nbdev={nbdev}")
                    self.assertTrue(np.allclose(this_l, that_l, equal_nan=True, rtol=1e-5, atol=1e-8), 
                                   f"Lower band mismatch at timeperiod={timeperiod}, nbdev={nbdev}")
                    
                    # 测试不同的上轨和下轨系数
                    this_u, this_m, this_l = this_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev+0.5, nbdevdn=nbdev-0.5)
                    that_u, that_m, that_l = that_BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev+0.5, nbdevdn=nbdev-0.5)
                    
                    self.assertTrue(np.allclose(this_u, that_u, equal_nan=True, rtol=1e-5, atol=1e-8), 
                                   f"Upper band mismatch at timeperiod={timeperiod}, nbdevup={nbdev+0.5}, nbdevdn={nbdev-0.5}")
                    self.assertTrue(np.allclose(this_l, that_l, equal_nan=True, rtol=1e-5, atol=1e-8), 
                                   f"Lower band mismatch at timeperiod={timeperiod}, nbdevup={nbdev+0.5}, nbdevdn={nbdev-0.5}")
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试全零输入
        close = np.zeros(100)
        this_u, this_m, this_l = this_BBANDS(close)
        that_u, that_m, that_l = that_BBANDS(close)
        
        self.assertTrue(np.allclose(this_u, that_u, equal_nan=True), "Mismatch with all zeros input")
        self.assertTrue(np.allclose(this_m, that_m, equal_nan=True), "Mismatch with all zeros input")
        self.assertTrue(np.allclose(this_l, that_l, equal_nan=True), "Mismatch with all zeros input")
        
        # 测试常量输入
        close = np.full(100, 50.0)
        this_u, this_m, this_l = this_BBANDS(close)
        that_u, that_m, that_l = that_BBANDS(close)
        
        self.assertTrue(np.allclose(this_u, that_u, equal_nan=True), "Mismatch with constant input")
        self.assertTrue(np.allclose(this_m, that_m, equal_nan=True), "Mismatch with constant input")
        self.assertTrue(np.allclose(this_l, that_l, equal_nan=True), "Mismatch with constant input")
        
        # 测试NaN和无穷大
        close = np.random.random(100)
        close[10] = np.nan
        close[20] = np.inf
        close[30] = -np.inf
        
        this_u, this_m, this_l = this_BBANDS(close)
        that_u, that_m, that_l = that_BBANDS(close)
        
        # 验证NaN和无穷大是否正确传播
        self.assertTrue(np.array_equal(np.isnan(this_u), np.isnan(that_u)), "NaN propagation mismatch")
        self.assertTrue(np.array_equal(np.isnan(this_m), np.isnan(that_m)), "NaN propagation mismatch")
        self.assertTrue(np.array_equal(np.isnan(this_l), np.isnan(that_l)), "NaN propagation mismatch")
        
        # 验证非NaN区域是否一致
        valid_mask = ~np.isnan(this_u) & ~np.isnan(that_u)
        self.assertTrue(np.allclose(this_u[valid_mask], that_u[valid_mask], equal_nan=True), 
                       "Mismatch in valid regions with NaN/inf")
        self.assertTrue(np.allclose(this_m[valid_mask], that_m[valid_mask], equal_nan=True), 
                       "Mismatch in valid regions with NaN/inf")
        self.assertTrue(np.allclose(this_l[valid_mask], that_l[valid_mask], equal_nan=True), 
                       "Mismatch in valid regions with NaN/inf")

if __name__ == '__main__':
    unittest.main()
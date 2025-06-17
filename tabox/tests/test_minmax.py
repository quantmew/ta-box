import numpy as np
import talib
from tabox import MINMAX


def test_minmax():
    # 生成测试数据
    np.random.seed(42)
    data = np.random.random(100)
    
    # 计算我们的实现
    min_result, max_result = MINMAX(data, timeperiod=30)
    
    # 计算talib的结果
    min_expected, max_expected = talib.MINMAX(data, timeperiod=30)
    
    # 比较结果
    np.testing.assert_array_equal(min_result, min_expected)
    np.testing.assert_array_equal(max_result, max_expected) 
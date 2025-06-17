import numpy as np
import talib
from tabox import MININDEX


def test_minindex():
    # 生成测试数据
    np.random.seed(42)
    data = np.random.random(100)
    
    # 计算我们的实现
    result = MININDEX(data, timeperiod=30)
    
    # 计算talib的结果
    expected = talib.MININDEX(data, timeperiod=30)
    
    # 比较结果
    np.testing.assert_array_equal(result, expected) 
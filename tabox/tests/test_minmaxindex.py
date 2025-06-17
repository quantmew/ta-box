import numpy as np
import talib
from tabox import MINMAXINDEX


def test_minmaxindex():
    # 生成测试数据
    np.random.seed(42)
    data = np.random.random(100)
    
    # 计算我们的实现
    minidx_result, maxidx_result = MINMAXINDEX(data, timeperiod=30)
    
    # 计算talib的结果
    minidx_expected, maxidx_expected = talib.MINMAXINDEX(data, timeperiod=30)
    
    # 比较结果
    np.testing.assert_array_equal(minidx_result, minidx_expected)
    np.testing.assert_array_equal(maxidx_result, maxidx_expected) 
import numpy as np
import cython

from .ta_utils import check_array, check_length4, check_begidx1

def TA_ADOSC_Lookback(optInFastPeriod: cython.Py_ssize_t, optInSlowPeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    pass

def TA_ADOSC(startIdx: cython.Py_ssize_t, endIdx: cython.Py_ssize_t, 
                 inHigh: cython.double[::1], inLow: cython.double[::1], 
                 inClose: cython.double[::1], inVolume: cython.double[::1],
                 optInFastPeriod: cython.Py_ssize_t, optInSlowPeriod: cython.Py_ssize_t,
                 outBegIdx: cython.Py_ssize_t[::1], outNBElement: cython.Py_ssize_t[::1],
                 outReal: cython.double[::1]) -> cython.int:
    return 0

def ADOSC(high, low, close, volume, fast_period=3, slow_period=10):
    """
    Chaikin A/D Oscillator (ADOSC)
    
    Parameters
    ----------
    high : array_like
        High prices
    low : array_like
        Low prices
    close : array_like
        Close prices
    volume : array_like
        Volume
    fast_period : int, optional
        Number of period for the fast MA (default: 3)
    slow_period : int, optional
        Number of period for the slow MA (default: 10)
        
    Returns
    -------
    out : ndarray
        The ADOSC values
    """
    # Check inputs
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    
    # Check array lengths
    check_length4(high, low, close, volume)
    
    # Check parameters
    # check_params(fast_period, 2, 100000, "fast_period")
    # check_params(slow_period, 2, 100000, "slow_period")
    
    # 计算lookback
    lookback = TA_ADOSC_Lookback(fast_period, slow_period)
    
    # 准备输出数组
    out = np.zeros(len(high) - lookback)
    out_beg_idx = np.array([0], dtype=np.int64)
    out_nb_element = np.array([0], dtype=np.int64)
    
    # 调用C函数
    ret = TA_ADOSC(0, len(high)-1, high, low, close, volume,
                   fast_period, slow_period,
                   out_beg_idx, out_nb_element, out)
    
    if ret != 0:
        raise Exception("Error calculating ADOSC")
        
    return out

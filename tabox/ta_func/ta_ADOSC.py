import numpy as np
import cython

from .ta_utils import check_array, check_length4, check_begidx4
from .ta_EMA import TA_EMA_Lookback

def TA_ADOSC_Lookback(optInFastPeriod: cython.Py_ssize_t, optInSlowPeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """Compute the lookback period for the ADOSC indicator"""
    if optInFastPeriod < optInSlowPeriod:
        slowest_period = optInSlowPeriod
    else:
        slowest_period = optInFastPeriod
    
    return TA_EMA_Lookback(slowest_period)

def TA_ADOSC(startIdx: cython.Py_ssize_t, endIdx: cython.Py_ssize_t, 
                 inHigh: cython.double[::1], inLow: cython.double[::1], 
                 inClose: cython.double[::1], inVolume: cython.double[::1],
                 optInFastPeriod: cython.Py_ssize_t, optInSlowPeriod: cython.Py_ssize_t,
                 outBegIdx: cython.Py_ssize_t[::1], outNBElement: cython.Py_ssize_t[::1],
                 outReal: cython.double[::1]) -> cython.int:
    """Compute the Chaikin A/D Oscillator"""
    # Check parameters
    if startIdx < 0:
        return -1
    if endIdx < 0 or endIdx < startIdx:
        return -1
    
    # Check default values for periods
    if optInFastPeriod == -1:
        optInFastPeriod = 3
    elif optInFastPeriod < 2 or optInFastPeriod > 100000:
        return -1
        
    if optInSlowPeriod == -1:
        optInSlowPeriod = 10
    elif optInSlowPeriod < 2 or optInSlowPeriod > 100000:
        return -1
    
    # Determine the slowest period
    if optInFastPeriod < optInSlowPeriod:
        slowest_period = optInSlowPeriod
    else:
        slowest_period = optInFastPeriod
    
    # Calculate lookback
    lookback = TA_EMA_Lookback(slowest_period)
    
    # Adjust startIdx
    if startIdx < lookback:
        startIdx = lookback
    
    # Check if there is enough data
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return 0
    
    # Initialize output
    outBegIdx[0] = startIdx
    today = startIdx - lookback
    
    # Calculate EMA k value
    fastk = 2.0 / (optInFastPeriod + 1)
    one_minus_fastk = 1.0 - fastk
    
    slowk = 2.0 / (optInSlowPeriod + 1)
    one_minus_slowk = 1.0 - slowk
    
    # Initialize AD and EMA
    ad = 0.0
    high = inHigh[today]
    low = inLow[today]
    close = inClose[today]
    tmp = high - low
    if tmp > 0.0:
        ad += (((close - low) - (high - close)) / tmp) * inVolume[today]
    today += 1
    
    fastEMA = ad
    slowEMA = ad
    
    # Initialize EMA
    while today < startIdx:
        high = inHigh[today]
        low = inLow[today]
        close = inClose[today]
        tmp = high - low
        if tmp > 0.0:
            ad += (((close - low) - (high - close)) / tmp) * inVolume[today]
        fastEMA = (fastk * ad) + (one_minus_fastk * fastEMA)
        slowEMA = (slowk * ad) + (one_minus_slowk * slowEMA)
        today += 1
    
    # Calculate the final result
    outIdx = 0
    while today <= endIdx:
        high = inHigh[today]
        low = inLow[today]
        close = inClose[today]
        tmp = high - low
        if tmp > 0.0:
            ad += (((close - low) - (high - close)) / tmp) * inVolume[today]
        fastEMA = (fastk * ad) + (one_minus_fastk * fastEMA)
        slowEMA = (slowk * ad) + (one_minus_slowk * slowEMA)
        outReal[outIdx] = fastEMA - slowEMA
        outIdx += 1
        today += 1
    
    outNBElement[0] = outIdx
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
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(high, low, close, volume)
    endidx = length - begidx - 1
    
    # Check parameters
    if fast_period < 2 or fast_period > 100000:
        raise ValueError("fast_period must be between 2 and 100000")
    if slow_period < 2 or slow_period > 100000:
        raise ValueError("slow_period must be between 2 and 100000")
    
    # Calculate lookback
    lookback = begidx + TA_ADOSC_Lookback(fast_period, slow_period)
    
    # Prepare output array
    out = np.full(length, np.nan, dtype=np.float64)
    out_beg_idx = np.array([0], dtype=np.intp)
    out_nb_element = np.array([0], dtype=np.intp)
    
    # Call C function
    ret = TA_ADOSC(0, endidx, 
                   high, low, close, volume,
                   fast_period, slow_period,
                   out_beg_idx, out_nb_element, out[lookback:])
    
    if ret != 0:
        raise Exception("Error calculating ADOSC")
        
    return out

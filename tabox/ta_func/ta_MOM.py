import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_MOM_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MOM_Lookback - Momentum Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 10
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return -1
    return optInTimePeriod


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MOM(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MOM - Momentum

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # Parameter check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:  # Default value handling
            optInTimePeriod = 10
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # If the start index is less than the lookback period, adjust the start index
    if startIdx < optInTimePeriod:
        startIdx = optInTimePeriod

    # Check if there is enough data for calculation
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    """
    Calculate momentum:
        Simply subtract the value of 'period' before the current value.
        
    The following is the change rate table implemented in TA-LIB:
        MOM     = (price - prevPrice)         [Momentum]
        ROC     = ((price/prevPrice)-1)*100   [Rate of Change]
        ROCP    = (price-prevPrice)/prevPrice [Rate of Change Percentage]
        ROCR    = (price/prevPrice)           [Rate of Change Ratio]
        ROCR100 = (price/prevPrice)*100       [Rate of Change Ratio 100 Ratio]
        
    Equivalent functions in other software:
        TA-Lib  |   Tradestation   |    Metastock         
        =================================================
        MOM     |   Momentum       |    ROC (Point)
        ROC     |   ROC            |    ROC (Percent)
        ROCP    |   PercentChange  |    -     
        ROCR    |   -              |    -
        ROCR100 |   -              |    MO
        
    MOM function is the only function that is not standardized, so it should be avoided for comparing different price time series.
    
    ROC and ROCP are centered at zero, and can have positive and negative values. The following are some equivalent relationships:
        ROC = ROCP/100 
            = ((price-prevPrice)/prevPrice)/100
            = ((price/prevPrice)-1)*100
            
    ROCR and ROCR100 are ratios centered at 1 and 100, and are always positive.
    """
    outIdx: cython.Py_ssize_t = 0
    inIdx: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - optInTimePeriod

    while inIdx <= endIdx:
        outReal[outIdx] = inReal[inIdx] - inReal[trailingIdx]
        outIdx += 1
        inIdx += 1
        trailingIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def MOM(real: np.ndarray, timeperiod: int = 10):
    """MOM(real[, timeperiod=10])

    Momentum (Overlap Studies)

    Momentum indicator calculates the difference between the current price and the price N periods ago, reflecting the speed of price change.

    Inputs:
        real: (any ndarray) Input sequence
    Parameters:
        timeperiod: 10 Number of periods
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MOM_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MOM(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

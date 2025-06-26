import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_ROCR100_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_ROCR100_Lookback - Rate of change ratio 100 scale Lookback

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
def TA_ROCR100(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 10
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Move up the start index if there is not enough initial data
    if startIdx < optInTimePeriod:
        startIdx = optInTimePeriod

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    """
    The interpretation of the rate of change varies widely depending
    which software and/or books you are refering to.

    The following is the table of Rate-Of-Change implemented in TA-LIB:
        MOM     = (price - prevPrice)         [Momentum]
        ROC     = ((price/prevPrice)-1)*100   [Rate of change]
        ROCP    = (price-prevPrice)/prevPrice [Rate of change Percentage]
        ROCR    = (price/prevPrice)           [Rate of change ratio]
        ROCR100 = (price/prevPrice)*100       [Rate of change ratio 100 Scale]

    Here are the equivalent function in other software:
        TA-Lib  |   Tradestation   |    Metastock         
        =================================================
        MOM     |   Momentum       |    ROC (Point)
        ROC     |   ROC            |    ROC (Percent)
        ROCP    |   PercentChange  |    -     
        ROCR    |   -              |    -
        ROCR100 |   -              |    MO

    The MOM function is the only one who is not normalized, and thus
    should be avoided for comparing different time serie of prices.

    ROC and ROCP are centered at zero and can have positive and negative
    value. Here are some equivalence:
        ROC = ROCP/100 
            = ((price-prevPrice)/prevPrice)/100
            = ((price/prevPrice)-1)*100

    ROCR and ROCR100 are ratio respectively centered at 1 and 100 and are
    always positive values.
    """

    # Calculate Rate of change Ratio: (price / prevPrice) * 100
    outIdx: cython.Py_ssize_t = 0
    inIdx: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - optInTimePeriod

    while inIdx <= endIdx:
        tempReal: cython.double = inReal[trailingIdx]
        trailingIdx += 1
        if tempReal != 0.0:
            outReal[outIdx] = (inReal[inIdx] / tempReal) * 100.0
        else:
            outReal[outIdx] = 0.0
        outIdx += 1
        inIdx += 1

    # Set output limits
    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def ROCR100(real: np.ndarray, timeperiod: int = 10) -> np.ndarray:
    """
    ROCR100(real[, timeperiod=10])

    Rate of change ratio 100 scale (Momentum Indicators)

    The ROCR100 is calculated as (price / prevPrice) * 100, where prevPrice is the price
    from 'timeperiod' periods ago.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 10 Number of periods to look back
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_ROCR100_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_ROCR100(
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

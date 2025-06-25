import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ROC(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_ROC - Rate of change : ((price/prevPrice)-1)*100

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if inReal is None:
            return TA_RetCode.TA_BAD_PARAM

        # 检查时间周期参数
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 10
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # 如果起始索引小于时间周期，调整起始索引
    if startIdx < optInTimePeriod:
        startIdx = optInTimePeriod

    # 确保有足够的数据进行计算
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # 计算ROC: ((price / prevPrice)-1)*100
    outIdx: cython.Py_ssize_t = 0
    inIdx: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - optInTimePeriod
    tempReal: cython.double

    while inIdx <= endIdx:
        tempReal = inReal[trailingIdx]
        trailingIdx += 1
        if tempReal != 0.0:
            outReal[outIdx] = ((inReal[inIdx] / tempReal) - 1.0) * 100.0
        else:
            outReal[outIdx] = 0.0
        outIdx += 1
        inIdx += 1

    # 设置输出范围
    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def TA_ROC_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    ROC_Lookback - Rate of change Lookback

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


def ROC(real: np.ndarray, timeperiod: int = 10) -> np.ndarray:
    """ROC(real[, timeperiod=10])

    Rate of change : ((price/prevPrice)-1)*100 (Momentum Indicators)

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 10 Number of periods
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_ROC_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_ROC(
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

import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

def TA_PLUS_DM_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_PLUS_DM_Lookback - Plus Directional Movement Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return -1

    if optInTimePeriod > 1:
        return (
            optInTimePeriod
            + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_PLUS_DM)
            - 1
        )
    else:
        return 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_PLUS_DM(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_PLUS_DM - Plus Directional Movement

    Input  = High, Low
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod: (From 1 to 100000)
        Number of period
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # 计算回溯期
    if optInTimePeriod > 1:
        lookbackTotal = (
            optInTimePeriod
            + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_PLUS_DM)
            - 1
        )
    else:
        lookbackTotal = 1

    # 调整起始索引以考虑回溯期
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 确保还有可计算的范围
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx = 0
    outBegIdx[0] = startIdx

    # 处理不需要平滑的情况
    if optInTimePeriod <= 1:
        today = startIdx - 1
        prevHigh = inHigh[today]
        prevLow = inLow[today]
        while today < endIdx:
            today += 1
            diffP = inHigh[today] - prevHigh  # 正方向差值
            prevHigh = inHigh[today]
            diffM = prevLow - inLow[today]  # 负方向差值
            prevLow = inLow[today]

            if diffP > 0 and diffP > diffM:
                outReal[outIdx] = diffP
            else:
                outReal[outIdx] = 0.0
            outIdx += 1

        outNBElement[0] = outIdx
        return TA_RetCode.TA_SUCCESS

    # 处理初始DM计算
    prevPlusDM = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    i = optInTimePeriod - 1
    while i > 0:
        today += 1
        diffP = inHigh[today] - prevHigh
        prevHigh = inHigh[today]
        diffM = prevLow - inLow[today]
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            prevPlusDM += diffP
        i -= 1

    # 跳过不稳定期
    i = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_PLUS_DM)
    while i > 0:
        today += 1
        diffP = inHigh[today] - prevHigh
        prevHigh = inHigh[today]
        diffM = prevLow - inLow[today]
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod) + diffP
        else:
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod)
        i -= 1

    # 写入第一个输出值
    outReal[0] = prevPlusDM
    outIdx = 1

    # 计算剩余周期
    while today < endIdx:
        today += 1
        diffP = inHigh[today] - prevHigh
        prevHigh = inHigh[today]
        diffM = prevLow - inLow[today]
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod) + diffP
        else:
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod)

        outReal[outIdx] = prevPlusDM
        outIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def PLUS_DM(high: np.ndarray, low: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    PLUS_DM(high, low[, timeperiod=14])

    Plus Directional Movement (Overlap Studies)

    The PLUS_DM is a measure of positive price movement, used in the calculation of the ADX indicator.

    Inputs:
        high: (any ndarray) High price series
        low: (any ndarray) Low price series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    check_timeperiod(timeperiod)

    if high.shape != low.shape:
        raise ValueError("High and low arrays must have the same shape")

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_PLUS_DM_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_PLUS_DM(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

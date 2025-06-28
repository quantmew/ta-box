import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

if not cython.compiled:
    from .ta_utility import TA_IS_ZERO


def TA_PLUS_DI_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_PLUS_DI_Lookback - Plus Directional Indicator Lookback

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
        return optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(
            TA_FuncUnstId.TA_FUNC_UNST_PLUS_DI
        )
    else:
        return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_PLUS_DI(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_PLUS_DI - Plus Directional Indicator

    Input  = High, Low, Close
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
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

    # 计算回溯期
    lookbackTotal: cython.Py_ssize_t = 0
    if optInTimePeriod > 1:
        lookbackTotal = optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(
            TA_FuncUnstId.TA_FUNC_UNST_PLUS_DI
        )
    else:
        lookbackTotal = 1

    # 调整起始索引
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 检查是否有数据需要处理
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx: cython.Py_ssize_t = 0
    outBegIdx[0] = startIdx

    # 处理不需要平滑的情况
    if optInTimePeriod <= 1:
        today: cython.Py_ssize_t = startIdx - 1
        prevHigh: cython.double = inHigh[today]
        prevLow: cython.double = inLow[today]
        prevClose: cython.double = inClose[today]
        diffP: cython.double = 0.0
        diffM: cython.double = 0.0
        tempReal: cython.double = 0.0
        tempReal2: cython.double = 0.0

        while today < endIdx:
            today += 1
            diffP = inHigh[today] - prevHigh  # 正增量
            diffM = prevLow - inLow[today]  # 负增量

            prevHigh = inHigh[today]
            prevLow = inLow[today]

            if diffP > 0 and diffP > diffM:
                # 情况1和3: +DM=diffP, -DM=0
                tempReal = 0.0
                # 计算真实范围
                tempReal = prevHigh - prevLow
                tempReal2 = abs(prevHigh - prevClose)
                if tempReal2 > tempReal:
                    tempReal = tempReal2
                tempReal2 = abs(prevLow - prevClose)
                if tempReal2 > tempReal:
                    tempReal = tempReal2

                if not TA_IS_ZERO(tempReal):
                    outReal[outIdx] = diffP / tempReal
                else:
                    outReal[outIdx] = 0.0
                outIdx += 1
            else:
                outReal[outIdx] = 0.0
                outIdx += 1

            prevClose = inClose[today]

        outNBElement[0] = outIdx
        return TA_RetCode.TA_SUCCESS

    # 处理初始DM和TR
    prevPlusDM = 0.0
    prevTR = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    prevClose = inClose[today]

    i: cython.Py_ssize_t = optInTimePeriod - 1
    while i > 0:
        today += 1
        diffP = inHigh[today] - prevHigh  # 正增量
        prevHigh = inHigh[today]

        diffM = prevLow - inLow[today]  # 负增量
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            # 情况1和3: +DM=diffP, -DM=0
            prevPlusDM += diffP

        # 计算真实范围
        tempReal = prevHigh - prevLow
        tempReal2 = abs(prevHigh - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2
        tempReal2 = abs(prevLow - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2

        prevTR += tempReal
        prevClose = inClose[today]
        i -= 1

    # 跳过不稳定期
    i = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_PLUS_DI) + 1
    while i > 0:
        today += 1
        diffP = inHigh[today] - prevHigh  # 正增量
        prevHigh = inHigh[today]

        diffM = prevLow - inLow[today]  # 负增量
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            # 情况1和3: +DM=diffP, -DM=0
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod) + diffP
        else:
            # 情况2,4,5和7
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod)

        # 计算真实范围
        tempReal = prevHigh - prevLow
        tempReal2 = abs(prevHigh - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2
        tempReal2 = abs(prevLow - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2

        prevTR = prevTR - (prevTR / optInTimePeriod) + tempReal
        prevClose = inClose[today]
        i -= 1

    # 计算第一个输出值
    if not TA_IS_ZERO(prevTR):
        outReal[0] = 100.0 * (prevPlusDM / prevTR)
    else:
        outReal[0] = 0.0
    outIdx = 1

    # 计算剩余值
    while today < endIdx:
        today += 1
        diffP = inHigh[today] - prevHigh  # 正增量
        prevHigh = inHigh[today]

        diffM = prevLow - inLow[today]  # 负增量
        prevLow = inLow[today]

        if diffP > 0 and diffP > diffM:
            # 情况1和3: +DM=diffP, -DM=0
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod) + diffP
        else:
            # 情况2,4,5和7
            prevPlusDM = prevPlusDM - (prevPlusDM / optInTimePeriod)

        # 计算真实范围
        tempReal = prevHigh - prevLow
        tempReal2 = abs(prevHigh - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2
        tempReal2 = abs(prevLow - prevClose)
        if tempReal2 > tempReal:
            tempReal = tempReal2

        prevTR = prevTR - (prevTR / optInTimePeriod) + tempReal
        prevClose = inClose[today]

        # 计算DI值
        if not TA_IS_ZERO(prevTR):
            outReal[outIdx] = 100.0 * (prevPlusDM / prevTR)
        else:
            outReal[outIdx] = 0.0
        outIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def PLUS_DI(
    inHigh: np.ndarray, inLow: np.ndarray, inClose: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """
    PLUS_DI(inHigh, inLow, inClose[, timeperiod=14])

    Plus Directional Indicator (Overlap Studies)

    The PLUS_DI is a technical indicator used to identify the direction of a price trend.
    It is calculated based on the relationship between high, low, and close prices.

    Inputs:
        inHigh: (ndarray) High prices
        inLow: (ndarray) Low prices
        inClose: (ndarray) Close prices
    Parameters:
        timeperiod: 14 Number of periods (From 1 to 100000)
    Outputs:
        real: Plus Directional Indicator values
    """
    inHigh = check_array(inHigh)
    inLow = check_array(inLow)
    inClose = check_array(inClose)
    check_timeperiod(timeperiod)

    length = inHigh.shape[0]
    startIdx = check_begidx1(inHigh)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_PLUS_DI_Lookback(timeperiod)

    outReal = np.full_like(inHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_PLUS_DI(
        0,
        endIdx,
        inHigh[startIdx:],
        inLow[startIdx:],
        inClose[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

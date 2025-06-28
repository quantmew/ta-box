import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MA import TA_MA, TA_MA_Lookback


def TA_MAVP_Lookback(
    optInMinPeriod: cython.int, optInMaxPeriod: cython.int, optInMAType: cython.int
) -> cython.Py_ssize_t:
    """
    TA_MAVP_Lookback - Moving average with variable period lookback

    Input:
        optInMinPeriod: (int) Minimum value for the period (From 2 to 100000)
        optInMaxPeriod: (int) Maximum value for the period (From 2 to 100000)
        optInMAType: (int) Type of Moving Average

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInMinPeriod == TA_INTEGER_DEFAULT:
            optInMinPeriod = 2
        elif optInMinPeriod < 2 or optInMinPeriod > 100000:
            return -1

        if optInMaxPeriod == TA_INTEGER_DEFAULT:
            optInMaxPeriod = 30
        elif optInMaxPeriod < 2 or optInMaxPeriod > 100000:
            return -1

        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return -1

    # 返回最大周期的回溯期
    return TA_MA_Lookback(optInMaxPeriod, optInMAType)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_MAVP(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    inPeriods: cython.double[::1],
    optInMinPeriod: cython.int,
    optInMaxPeriod: cython.int,
    optInMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MAVP - Moving average with variable period

    Input  = double, double
    Output = double

    Optional Parameters
    -------------------
    optInMinPeriod:(From 2 to 100000)
       Value less than minimum will be changed to Minimum period
    optInMaxPeriod:(From 2 to 100000)
       Value higher than maximum will be changed to Maximum period
    optInMAType:
       Type of Moving Average
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if optInMinPeriod == TA_INTEGER_DEFAULT:
            optInMinPeriod = 2
        elif optInMinPeriod < 2 or optInMinPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInMaxPeriod == TA_INTEGER_DEFAULT:
            optInMaxPeriod = 30
        elif optInMaxPeriod < 2 or optInMaxPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return TA_RetCode.TA_BAD_PARAM

        if inReal is None or inPeriods is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    length: cython.Py_ssize_t = endIdx - startIdx + 1
    tempBuffer: cython.double[::1] = np.full(length, np.nan, dtype=np.double)
    tempPeriodBuffer: cython.int[::1] = np.zeros(length, dtype=np.int32)

    retCode: cython.int
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    outputSize: cython.Py_ssize_t
    tempInt: cython.int
    curPeriod: cython.int
    outBegIdx1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNbElement1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    # 确定计算至少一个输出所需的最小价格柱数
    lookbackTotal: cython.Py_ssize_t = TA_MAVP_Lookback(
        optInMinPeriod, optInMaxPeriod, optInMAType
    )

    # 如果起始索引小于回溯期，则上移起始索引
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 确保仍有数据可评估
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # 计算确切的输出大小
    if lookbackTotal > startIdx:
        tempInt = lookbackTotal
    else:
        tempInt = startIdx
    if tempInt > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    outputSize = endIdx - tempInt + 1

    # 复制调用者的周期数组到本地缓冲区，同时截断到最小/最大值
    for i in range(outputSize):
        tempInt = int(inPeriods[startIdx + i])
        if tempInt < optInMinPeriod:
            tempInt = optInMinPeriod
        elif tempInt > optInMaxPeriod:
            tempInt = optInMaxPeriod
        tempPeriodBuffer[i] = tempInt

    # 处理输入的每个元素
    for i in range(outputSize):
        curPeriod = tempPeriodBuffer[i]
        if curPeriod != 0:
            # 需要计算MA
            retCode = TA_MA(
                startIdx,
                endIdx,
                inReal,
                curPeriod,
                optInMAType,
                outBegIdx1,
                outNbElement1,
                tempBuffer,
            )

            if retCode != TA_RetCode.TA_SUCCESS:
                outBegIdx[0] = 0
                outNBElement[0] = 0
                return retCode

            outReal[i] = tempBuffer[i]
            # 为相同周期的元素填充结果，避免重复计算
            for j in range(i + 1, outputSize):
                if tempPeriodBuffer[j] == curPeriod:
                    tempPeriodBuffer[j] = 0  # 标记以避免重新计算
                    outReal[j] = tempBuffer[j]

    outBegIdx[0] = startIdx
    outNBElement[0] = outputSize
    return TA_RetCode.TA_SUCCESS


def MAVP(
    real: np.ndarray,
    periods: np.ndarray,
    minperiod: int = 2,
    maxperiod: int = 30,
    matype: int = 0,
) -> np.ndarray:
    """MAVP(real, periods[, minperiod=2, maxperiod=30, matype=0])

    Moving Average with Variable Period (Overlap Studies)

    The MAVP calculates a moving average where the period for each point
    is given by the 'periods' input array, clamped to the min and max periods.

    Inputs:
        real: (any ndarray) Input series
        periods: (any ndarray) Periods for each point
    Parameters:
        minperiod: 2 Minimum allowed period
        maxperiod: 30 Maximum allowed period
        matype: 0 Type of moving average (0=SMA, 1=EMA, etc.)
    Outputs:
        real
    """
    real = check_array(real)
    periods = check_array(periods)

    # 检查周期参数
    check_timeperiod(minperiod)
    check_timeperiod(maxperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MAVP_Lookback(
        minperiod, maxperiod, matype
    )

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MAVP(
        0,
        endIdx,
        real[startIdx:],
        periods[startIdx:],
        minperiod,
        maxperiod,
        matype,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

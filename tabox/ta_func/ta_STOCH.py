import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MA import TA_MA, TA_MA_Lookback


def TA_STOCH_Lookback(
    optInFastK_Period: cython.int,
    optInSlowK_Period: cython.int,
    optInSlowK_MAType: cython.int,
    optInSlowD_Period: cython.int,
    optInSlowD_MAType: cython.int,
) -> cython.Py_ssize_t:
    """
    TA_STOCH_Lookback - Stochastic Lookback

    Input:
        optInFastK_Period: (int) Time period for building the Fast-K line (From 1 to 100000)
        optInSlowK_Period: (int) Smoothing for making the Slow-K line (From 1 to 100000)
        optInSlowK_MAType: (int) Type of Moving Average for Slow-K
        optInSlowD_Period: (int) Smoothing for making the Slow-D line (From 1 to 100000)
        optInSlowD_MAType: (int) Type of Moving Average for Slow-D

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return -1

        if optInSlowK_Period == TA_INTEGER_DEFAULT:
            optInSlowK_Period = 3
        elif optInSlowK_Period < 1 or optInSlowK_Period > 100000:
            return -1

        if optInSlowK_MAType == TA_INTEGER_DEFAULT:
            optInSlowK_MAType = 0
        elif optInSlowK_MAType < 0 or optInSlowK_MAType > 8:
            return -1

        if optInSlowD_Period == TA_INTEGER_DEFAULT:
            optInSlowD_Period = 3
        elif optInSlowD_Period < 1 or optInSlowD_Period > 100000:
            return -1

        if optInSlowD_MAType == TA_INTEGER_DEFAULT:
            optInSlowD_MAType = 0
        elif optInSlowD_MAType < 0 or optInSlowD_MAType > 8:
            return -1

    # 计算各部分的回溯期并求和
    lookbackK = optInFastK_Period - 1
    lookbackKSlow = TA_MA_Lookback(optInSlowK_Period, optInSlowK_MAType)
    lookbackDSlow = TA_MA_Lookback(optInSlowD_Period, optInSlowD_MAType)
    return lookbackK + lookbackKSlow + lookbackDSlow


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_STOCH(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInFastK_Period: cython.int,
    optInSlowK_Period: cython.int,
    optInSlowK_MAType: cython.int,
    optInSlowD_Period: cython.int,
    optInSlowD_MAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outSlowK: cython.double[::1],
    outSlowD: cython.double[::1],
) -> cython.int:
    """TA_STOCH - Stochastic

    Input  = High, Low, Close (double)
    Output = SlowK, SlowD (double)

    Optional Parameters
    -------------------
    optInFastK_Period:(From 1 to 100000)
       Time period for building the Fast-K line
    optInSlowK_Period:(From 1 to 100000)
       Smoothing for making the Slow-K line. Usually set to 3
    optInSlowK_MAType:
       Type of Moving Average for Slow-K
    optInSlowD_Period:(From 1 to 100000)
       Smoothing for making the Slow-D line
    optInSlowD_MAType:
       Type of Moving Average for Slow-D
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outSlowK is None or outSlowD is None:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowK_Period == TA_INTEGER_DEFAULT:
            optInSlowK_Period = 3
        elif optInSlowK_Period < 1 or optInSlowK_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowK_MAType == TA_INTEGER_DEFAULT:
            optInSlowK_MAType = 0
        elif optInSlowK_MAType < 0 or optInSlowK_MAType > 8:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowD_Period == TA_INTEGER_DEFAULT:
            optInSlowD_Period = 3
        elif optInSlowD_Period < 1 or optInSlowD_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowD_MAType == TA_INTEGER_DEFAULT:
            optInSlowD_MAType = 0
        elif optInSlowD_MAType < 0 or optInSlowD_MAType > 8:
            return TA_RetCode.TA_BAD_PARAM

    # 计算所需的回溯期
    lookbackK: cython.Py_ssize_t = optInFastK_Period - 1
    lookbackKSlow: cython.Py_ssize_t = TA_MA_Lookback(optInSlowK_Period, optInSlowK_MAType)
    lookbackDSlow: cython.Py_ssize_t = TA_MA_Lookback(optInSlowD_Period, optInSlowD_MAType)
    lookbackTotal: cython.Py_ssize_t = lookbackK + lookbackKSlow + lookbackDSlow

    # 调整起始索引以确保有足够的数据
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # 计算Fast-K所需的临时缓冲区大小
    tempBuffer_size: cython.Py_ssize_t = endIdx - (startIdx - lookbackTotal) + 1
    tempBuffer: cython.double[::1] = np.full(tempBuffer_size, np.nan, dtype=np.double)

    outIdx: cython.Py_ssize_t = 0
    trailingIdx: cython.Py_ssize_t = startIdx - lookbackTotal
    today: cython.Py_ssize_t = trailingIdx + lookbackK
    lowestIdx: cython.Py_ssize_t = -1
    highestIdx: cython.Py_ssize_t = -1
    diff: cython.double = 0.0
    highest: cython.double = 0.0
    lowest: cython.double = 0.0

    tmp: cython.double = 0.0
    i: cython.Py_ssize_t = 0

    # 计算Fast-K值
    while today <= endIdx:
        # 设置最低价格
        tmp = inLow[today]
        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inLow[lowestIdx]
            i = lowestIdx
            while i <= today:
                tmp = inLow[i]
                if tmp < lowest:
                    lowestIdx = i
                    lowest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp
            diff = (highest - lowest) / 100.0

        # 设置最高价格
        tmp = inHigh[today]
        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inHigh[highestIdx]
            i = highestIdx
            while i <= today:
                tmp = inHigh[i]
                if tmp > highest:
                    highestIdx = i
                    highest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp >= highest:
            highestIdx = today
            highest = tmp
            diff = (highest - lowest) / 100.0

        # 计算随机指标值
        if diff != 0.0:
            tempBuffer[outIdx] = (inClose[today] - lowest) / diff
        else:
            tempBuffer[outIdx] = 0.0
        outIdx += 1
        trailingIdx += 1
        today += 1

    # 计算Slow-K (对Fast-K进行移动平均)
    outBegIdx1 = np.zeros(1, dtype=np.intp)
    outNBElement1 = np.zeros(1, dtype=np.intp)
    retCode = TA_MA(
        0,
        outIdx - 1,
        tempBuffer,
        optInSlowK_Period,
        optInSlowK_MAType,
        outBegIdx1,
        outNBElement1,
        tempBuffer,
    )

    if retCode != TA_RetCode.TA_SUCCESS or outNBElement1[0] == 0:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    # 计算Slow-D (对Slow-K进行移动平均)
    outBegIdx2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    retCode = TA_MA(
        0,
        outNBElement1[0] - 1,
        tempBuffer,
        optInSlowD_Period,
        optInSlowD_MAType,
        outBegIdx2,
        outNBElement2,
        outSlowD,
    )

    # 将Slow-K复制到输出数组
    slowK_start: cython.Py_ssize_t = lookbackDSlow
    slowK_count: cython.Py_ssize_t = outNBElement2[0]
    for i in range(slowK_count):
        outSlowK[i] = tempBuffer[slowK_start + i]

    if retCode != TA_RetCode.TA_SUCCESS:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    outBegIdx[0] = startIdx
    outNBElement[0] = outNBElement2[0]
    return TA_RetCode.TA_SUCCESS


def STOCH(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """STOCH(high, low, close[, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0])

    Stochastic Oscillator (Momentum Indicators)

    The Stochastic Oscillator is calculated as:
    FASTK = 100 * (close - lowest low) / (highest high - lowest low)
    SLOWK = MA of FASTK
    SLOWD = MA of SLOWK

    Inputs:
        high: (any ndarray) High price series
        low: (any ndarray) Low price series
        close: (any ndarray) Close price series
    Parameters:
        fastk_period: 5 Time period for Fast-K
        slowk_period: 3 Time period for Slow-K MA
        slowk_matype: 0 Type of MA for Slow-K (0=SMA, 1=EMA, etc.)
        slowd_period: 3 Time period for Slow-D MA
        slowd_matype: 0 Type of MA for Slow-D (0=SMA, 1=EMA, etc.)
    Outputs:
        slowk, slowd
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    if high.shape[0] != low.shape[0] or high.shape[0] != close.shape[0]:
        raise ValueError("Input arrays must have the same length")

    check_timeperiod(fastk_period)
    check_timeperiod(slowk_period)
    check_timeperiod(slowd_period)

    length = high.shape[0]
    startIdx = check_begidx1(high)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_STOCH_Lookback(
        fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype
    )

    outSlowK = np.full_like(high, np.nan)
    outSlowD = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_STOCH(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        fastk_period,
        slowk_period,
        slowk_matype,
        slowd_period,
        slowd_matype,
        outBegIdx,
        outNBElement,
        outSlowK[lookback:],
        outSlowD[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outSlowK, outSlowD
    return outSlowK, outSlowD

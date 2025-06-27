import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MA import TA_MA, TA_MA_Lookback


if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT


def TA_PPO_Lookback(
    optInFastPeriod: cython.int, optInSlowPeriod: cython.int, optInMAType: cython.int
) -> cython.Py_ssize_t:
    """
    TA_PPO_Lookback - Percentage Price Oscillator Lookback

    Input:
        optInFastPeriod: (int) Number of period for the fast MA (From 2 to 100000)
        optInSlowPeriod: (int) Number of period for the slow MA (From 2 to 100000)
        optInMAType: (int) Type of Moving Average

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInFastPeriod == TA_INTEGER_DEFAULT:
            optInFastPeriod = 12
        elif optInFastPeriod < 2 or optInFastPeriod > 100000:
            return -1

        if optInSlowPeriod == TA_INTEGER_DEFAULT:
            optInSlowPeriod = 26
        elif optInSlowPeriod < 2 or optInSlowPeriod > 100000:
            return -1

        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return -1

    return TA_MA_Lookback(max(optInSlowPeriod, optInFastPeriod), optInMAType)


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_PPO(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInFastPeriod: cython.int,
    optInSlowPeriod: cython.int,
    optInMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_PPO - Percentage Price Oscillator

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInFastPeriod:(From 2 to 100000)
       Number of period for the fast MA
    optInSlowPeriod:(From 2 to 100000)
       Number of period for the slow MA
    optInMAType:
       Type of Moving Average
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if optInFastPeriod == TA_INTEGER_DEFAULT:
            optInFastPeriod = 12
        elif optInFastPeriod < 2 or optInFastPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowPeriod == TA_INTEGER_DEFAULT:
            optInSlowPeriod = 26
        elif optInSlowPeriod < 2 or optInSlowPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return TA_RetCode.TA_BAD_PARAM

        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    length = endIdx - startIdx + 1
    tempBuffer = np.full(length, np.nan, dtype=np.double)

    retCode: cython.int
    tempInteger: cython.int
    outBegIdx1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNbElement1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outBegIdx2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNbElement2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    tempReal: cython.double

    # 确保慢周期大于快周期，如果不是则交换它们
    if optInSlowPeriod < optInFastPeriod:
        tempInteger = optInSlowPeriod
        optInSlowPeriod = optInFastPeriod
        optInFastPeriod = tempInteger

    # 计算快速MA到临时缓冲区
    retCode = TA_MA(
        startIdx,
        endIdx,
        inReal,
        optInFastPeriod,
        optInMAType,
        outBegIdx2,
        outNbElement2,
        tempBuffer,
    )

    if retCode == TA_RetCode.TA_SUCCESS:
        # 计算慢速MA到输出数组
        retCode = TA_MA(
            startIdx,
            endIdx,
            inReal,
            optInSlowPeriod,
            optInMAType,
            outBegIdx1,
            outNbElement1,
            outReal,
        )

        if retCode == TA_RetCode.TA_SUCCESS:
            tempInteger = outBegIdx1[0] - outBegIdx2[0]
            # 计算百分比振荡器: ((快速MA - 慢速MA)/慢速MA) * 100
            for i, j in zip(
                range(outNbElement1[0]),
                range(tempInteger, tempInteger + outNbElement1[0]),
            ):
                tempReal = outReal[i]
                if not TA_IS_ZERO(tempReal):
                    outReal[i] = ((tempBuffer[j] - tempReal) / tempReal) * 100.0
                else:
                    outReal[i] = 0.0

            outBegIdx[0] = outBegIdx1[0]
            outNBElement[0] = outNbElement1[0]

    if retCode != TA_RetCode.TA_SUCCESS:
        outBegIdx[0] = 0
        outNBElement[0] = 0

    return retCode


def PPO(
    real: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0
) -> np.ndarray:
    """PPO(real[, fastperiod=12, slowperiod=26, matype=0])

    Percentage Price Oscillator (Overlap Studies)

    The PPO is calculated by subtracting a slow moving average from a fast moving average,
    then dividing the result by the slow moving average and multiplying by 100.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        fastperiod: 12 Number of periods for the fast MA
        slowperiod: 26 Number of periods for the slow MA
        matype: 0 Type of moving average (0=SMA, 1=EMA, etc.)
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(fastperiod)
    check_timeperiod(slowperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_PPO_Lookback(
        fastperiod, slowperiod, matype
    )

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_PPO(
        0,
        endIdx,
        real[startIdx:],
        fastperiod,
        slowperiod,
        matype,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

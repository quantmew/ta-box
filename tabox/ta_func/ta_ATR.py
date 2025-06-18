import cython
import numpy as np
from .ta_utils import check_array, check_begidx3, check_length3, make_double_array
from .ta_TRANGE import TA_TRANGE
from .ta_SMA import TA_INT_SMA
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId

def TA_ATR_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    unstable_period = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_ATR)
    return optInTimePeriod + unstable_period


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ATR(
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
    # 计算总lookback周期
    lookbackTotal: cython.Py_ssize_t = TA_ATR_Lookback(optInTimePeriod)
    
    # 调整起始索引
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    if optInTimePeriod <= 1:
        return TA_TRANGE(startIdx, endIdx, inHigh, inLow, inClose, outReal)

    # 计算临时缓冲区大小，与C语言一致
    buffer_size = lookbackTotal + (endIdx - startIdx) + 1
    tempBuffer = np.zeros(buffer_size, dtype=float)
    prevATRTemp = np.zeros(1, dtype=float)
    outBegIdx1 = np.zeros(1, dtype=np.intp)
    outNBElement1 = np.zeros(1, dtype=np.intp)

    # 计算真实范围(TRANGE)
    tr_start = startIdx - lookbackTotal + 1
    retCode = TA_TRANGE(tr_start, endIdx, inHigh, inLow, inClose, tempBuffer)
    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    # 计算第一个ATR值，使用SMA
    retCode = TA_INT_SMA(
        optInTimePeriod - 1,
        optInTimePeriod - 1,
        tempBuffer,
        optInTimePeriod,
        outBegIdx1,
        outNBElement1,
        prevATRTemp,
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    prevATR = prevATRTemp[0]

    # 获取不稳定周期，修正之前的硬编码问题
    unstablePeriod = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_ATR)
    
    # 处理不稳定周期
    today = optInTimePeriod
    outIdx = 0
    while unstablePeriod != 0:
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        unstablePeriod -= 1

    # 写入第一个ATR值
    outReal[outIdx] = prevATR
    outIdx += 1

    # 计算剩余的ATR值
    nbATR = (endIdx - startIdx) + 1
    while nbATR - 1 != 0:
        nbATR -= 1
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        outReal[outIdx] = prevATR
        outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def ATR(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """ATR(high, low, close[, timeperiod=?])

    Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """

    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    startIdx = check_begidx3(high, low, close)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_ATR_Lookback(timeperiod)
    outreal = make_double_array(length, lookback)
    
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    TA_ATR(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outreal[lookback:],
    )
    return outreal
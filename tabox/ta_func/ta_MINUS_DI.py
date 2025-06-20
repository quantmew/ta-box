import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT, TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from math import fabs

def TRUE_RANGE(
    th: cython.double, tl: cython.double, yc: cython.double
) -> cython.double:
    """
    Calculate the True Range

    Input:
        th: Today's high
        tl: Today's low
        yc: Yesterday's close

    Output:
        (float) True Range
    """
    tr = th - tl
    temp_real2 = fabs(th - yc)
    if temp_real2 > tr:
        tr = temp_real2
    temp_real2 = fabs(tl - yc)
    if temp_real2 > tr:
        tr = temp_real2
    return tr

def TA_MINUS_DI_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MINUS_DI_Lookback - Minus Directional Indicator Lookback
    
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
        return optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DI)
    else:
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MINUS_DI(
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
    TA_MINUS_DI - Minus Directional Indicator
    
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
        
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
    
    # 计算回溯期
    if optInTimePeriod > 1:
        lookbackTotal = optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DI)
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
    
    # 初始化变量
    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t
    prevHigh: cython.double
    prevLow: cython.double
    prevClose: cython.double
    prevMinusDM: cython.double
    prevTR: cython.double
    diffP: cython.double  # Plus Delta
    diffM: cython.double  # Minus Delta
    tempReal: cython.double
    
    # 处理不需要平滑的情况
    if optInTimePeriod <= 1:
        outBegIdx[0] = startIdx
        today = startIdx - 1
        prevHigh = inHigh[today]
        prevLow = inLow[today]
        prevClose = inClose[today]
        
        while today < endIdx:
            today += 1
            tempReal = inHigh[today]
            diffP = tempReal - prevHigh  # Plus Delta
            prevHigh = tempReal
            
            tempReal = inLow[today]
            diffM = prevLow - tempReal  # Minus Delta
            prevLow = tempReal
            
            if (diffM > 0) and (diffP < diffM):
                # 情况2和4: +DM=0, -DM=diffM
                tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
                if TA_IS_ZERO(tr):
                    outReal[outIdx] = 0.0
                else:
                    outReal[outIdx] = diffM / tr
            else:
                outReal[outIdx] = 0.0
            
            outIdx += 1
            prevClose = inClose[today]
        
        outNBElement[0] = outIdx
        return TA_RetCode.TA_SUCCESS
    
    # 处理初始DM和TR
    outBegIdx[0] = today = startIdx
    prevMinusDM = 0.0
    prevTR = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    prevClose = inClose[today]
    i = optInTimePeriod - 1
    
    while i > 0:
        i -= 1
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal
        
        if (diffM > 0) and (diffP < diffM):
            # 情况2和4: +DM=0, -DM=diffM
            prevMinusDM += diffM
        
        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR += tr
        prevClose = inClose[today]
    
    # 处理不稳定期
    unstable_period = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DI)
    i = unstable_period + 1
    while i > 0:
        i -= 1
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal
        
        if (diffM > 0) and (diffP < diffM):
            # 情况2和4: +DM=0, -DM=diffM
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod) + diffM
        else:
            # 情况1,3,5和7
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod)
        
        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR = prevTR - (prevTR / optInTimePeriod) + tr
        prevClose = inClose[today]
    
    # 计算第一个输出值
    if not TA_IS_ZERO(prevTR):
        outReal[0] = 100.0 * (prevMinusDM / prevTR)
    else:
        outReal[0] = 0.0
    outIdx = 1
    
    # 计算剩余的输出值
    while today < endIdx:
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal
        
        if (diffM > 0) and (diffP < diffM):
            # 情况2和4: +DM=0, -DM=diffM
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod) + diffM
        else:
            # 情况1,3,5和7
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod)
        
        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR = prevTR - (prevTR / optInTimePeriod) + tr
        prevClose = inClose[today]
        
        if not TA_IS_ZERO(prevTR):
            outReal[outIdx] = 100.0 * (prevMinusDM / prevTR)
        else:
            outReal[outIdx] = 0.0
        
        outIdx += 1
    
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def MINUS_DI(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """
    MINUS_DI(high, low, close[, timeperiod=14])
    
    Minus Directional Indicator (Overlap Studies)
    
    The MINUS_DI is used in the Directional Movement Index (DMI) system
    to measure the strength of downward price movement.
    
    Inputs:
        high: (any ndarray) High price series
        low: (any ndarray) Low price series
        close: (any ndarray) Close price series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    
    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("Input arrays must have the same shape")
    
    check_timeperiod(timeperiod)
    
    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MINUS_DI_Lookback(timeperiod)
    
    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    retCode = TA_MINUS_DI(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
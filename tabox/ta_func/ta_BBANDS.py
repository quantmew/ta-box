import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode, TA_INTEGER_DEFAULT, TA_MAType
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MA import TA_MA, TA_MA_Lookback
from .ta_STDDEV import TA_STDDEV, INT_stddev_using_precalc_ma
from .ta_defs import TA_INTEGER_DEFAULT, TA_REAL_DEFAULT, TA_REAL_MIN, TA_REAL_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_BBANDS_Lookback(
    optInTimePeriod: cython.Py_ssize_t,
    optInNbDevUp: cython.double,
    optInNbDevDn: cython.double,
    optInMAType: cython.int
) -> cython.Py_ssize_t:
    """
    TA_BBANDS_Lookback - Bollinger Bands Lookback
    
    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)
        optInNbDevUp: (double) Deviation multiplier for upper band
        optInNbDevDn: (double) Deviation multiplier for lower band
        optInMAType: (int) Type of Moving Average
    
    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 5
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
            
        if optInNbDevUp == TA_REAL_DEFAULT:
            optInNbDevUp = 2.0
        elif optInNbDevUp < TA_REAL_MIN or optInNbDevUp > TA_REAL_MAX:
            return -1
            
        if optInNbDevDn == TA_REAL_DEFAULT:
            optInNbDevDn = 2.0
        elif optInNbDevDn < TA_REAL_MIN or optInNbDevDn > TA_REAL_MAX:
            return -1
            
        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return -1
    
    # The lookback is driven by the middle band moving average
    return TA_MA_Lookback(optInTimePeriod, optInMAType)


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_BBANDS(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInNbDevUp: cython.double,
    optInNbDevDn: cython.double,
    optInMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outRealUpperBand: cython.double[::1],
    outRealMiddleBand: cython.double[::1],
    outRealLowerBand: cython.double[::1]
) -> cython.int:
    """Internal BBANDS implementation without parameter checks"""
    retCode: cython.int
    i: cython.Py_ssize_t
    tempReal: cython.double
    tempReal2: cython.double
    
    # Identify temporary buffers
    tempBuffer1 = outRealMiddleBand
    tempBuffer2 = outRealLowerBand
    
    # Calculate the middle band (moving average)
    retCode = TA_MA(
        startIdx, endIdx, inReal, optInTimePeriod, optInMAType,
        outBegIdx, outNBElement, tempBuffer1
    )
    
    if retCode != TA_RetCode.TA_SUCCESS or outNBElement[0] == 0:
        outNBElement[0] = 0
        return retCode
    
    # Calculate the standard deviation into tempBuffer2
    if optInMAType == TA_MAType.TA_MAType_SMA:  # TA_MAType_SMA
        # Optimized version for SMA
        INT_stddev_using_precalc_ma(inReal, tempBuffer1, outBegIdx[0], outNBElement[0], optInTimePeriod, tempBuffer2)
    else:
        # Use standard STDDEV calculation
        from .ta_STDDEV import TA_STDDEV
        stddev_startIdx = outBegIdx[0]
        stddev_endIdx = endIdx
        stddev_optInTimePeriod = optInTimePeriod
        stddev_optInNbDev = 1.0
        
        stddev_outBegIdx = np.zeros(1, dtype=np.intp)
        stddev_outNBElement = np.zeros(1, dtype=np.intp)
        stddev_outReal = np.full_like(inReal, 0.0)
        
        retCode = TA_STDDEV(
            stddev_startIdx, stddev_endIdx, inReal,
            stddev_optInTimePeriod, stddev_optInNbDev,
            stddev_outBegIdx, stddev_outNBElement, stddev_outReal
        )
        
        if retCode != TA_RetCode.TA_SUCCESS:
            outNBElement[0] = 0
            return retCode
        
        # Copy stddev results to tempBuffer2
        for i in range(stddev_outNBElement[0]):
            tempBuffer2[i] = stddev_outReal[i + startIdx - stddev_startIdx]
    
    # Calculate upper and lower bands
    if optInNbDevUp == optInNbDevDn:
        if optInNbDevUp == 1.0:
            for i in range(outNBElement[0]):
                tempReal = tempBuffer2[i]
                tempReal2 = outRealMiddleBand[i]
                outRealUpperBand[i] = tempReal2 + tempReal
                outRealLowerBand[i] = tempReal2 - tempReal
        else:
            mult = optInNbDevUp
            for i in range(outNBElement[0]):
                tempReal = tempBuffer2[i] * mult
                tempReal2 = outRealMiddleBand[i]
                outRealUpperBand[i] = tempReal2 + tempReal
                outRealLowerBand[i] = tempReal2 - tempReal
    elif optInNbDevUp == 1.0:
        for i in range(outNBElement[0]):
            tempReal = tempBuffer2[i]
            tempReal2 = outRealMiddleBand[i]
            outRealUpperBand[i] = tempReal2 + tempReal
            outRealLowerBand[i] = tempReal2 - (tempReal * optInNbDevDn)
    elif optInNbDevDn == 1.0:
        for i in range(outNBElement[0]):
            tempReal = tempBuffer2[i]
            tempReal2 = outRealMiddleBand[i]
            outRealLowerBand[i] = tempReal2 - tempReal
            outRealUpperBand[i] = tempReal2 + (tempReal * optInNbDevUp)
    else:
        for i in range(outNBElement[0]):
            tempReal = tempBuffer2[i]
            tempReal2 = outRealMiddleBand[i]
            outRealUpperBand[i] = tempReal2 + (tempReal * optInNbDevUp)
            outRealLowerBand[i] = tempReal2 - (tempReal * optInNbDevDn)
    
    return TA_RetCode.TA_SUCCESS


def TA_BBANDS(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInNbDevUp: cython.double,
    optInNbDevDn: cython.double,
    optInMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outRealUpperBand: cython.double[::1],
    outRealMiddleBand: cython.double[::1],
    outRealLowerBand: cython.double[::1]
) -> cython.int:
    """TA_BBANDS - Bollinger Bands
    
    Input  = double
    Output = double, double, double
    
    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    optInNbDevUp:(From TA_REAL_MIN to TA_REAL_MAX)
       Deviation multiplier for upper band
    optInNbDevDn:(From TA_REAL_MIN to TA_REAL_MAX)
       Deviation multiplier for lower band
    optInMAType:
       Type of Moving Average
    """
    # parameters check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal is None or outRealUpperBand is None or outRealMiddleBand is None or outRealLowerBand is None:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInTimePeriod == 0:  # 默认值处理
            optInTimePeriod = 5
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInNbDevUp == TA_REAL_DEFAULT:
            optInNbDevUp = 2.0
        elif optInNbDevUp < TA_REAL_MIN or optInNbDevUp > TA_REAL_MAX:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInNbDevDn == TA_REAL_DEFAULT:
            optInNbDevDn = 2.0
        elif optInNbDevDn < TA_REAL_MIN or optInNbDevDn > TA_REAL_MAX:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInMAType == TA_INTEGER_DEFAULT:
            optInMAType = 0
        elif optInMAType < 0 or optInMAType > 8:
            return TA_RetCode.TA_BAD_PARAM
    
    # Call internal implementation
    return TA_INT_BBANDS(
        startIdx, endIdx, inReal, optInTimePeriod, optInNbDevUp, optInNbDevDn, optInMAType,
        outBegIdx, outNBElement, outRealUpperBand, outRealMiddleBand, outRealLowerBand
    )


def BBANDS(
    real: np.ndarray,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BBANDS(real[, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0])
    
    Bollinger Bands (Overlap Studies)
    
    Inputs:
        real: (any ndarray) Input array
    Parameters:
        timeperiod: 5
        nbdevup: 2.0
        nbdevdn: 2.0
        matype: 0 (SMA)
    Outputs:
        upperband, middleband, lowerband
    """
    real = check_array(real)
    check_timeperiod(timeperiod)
    
    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_BBANDS_Lookback(timeperiod, nbdevup, nbdevdn, matype)
    
    outUpperBand = np.full_like(real, np.nan)
    outMiddleBand = np.full_like(real, np.nan)
    outLowerBand = np.full_like(real, np.nan)
    
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    retCode = TA_BBANDS(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        nbdevup,
        nbdevdn,
        matype,
        outBegIdx,
        outNBElement,
        outUpperBand[lookback:],
        outMiddleBand[lookback:],
        outLowerBand[lookback:]
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outUpperBand, outMiddleBand, outLowerBand
    
    return outUpperBand, outMiddleBand, outLowerBand
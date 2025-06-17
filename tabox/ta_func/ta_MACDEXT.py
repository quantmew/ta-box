import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import *
from .ta_MA import TA_MA, TA_MA_Lookback

def TA_MACDEXT_Lookback(optInFastPeriod: cython.int, optInFastMAType: cython.int,
                       optInSlowPeriod: cython.int, optInSlowMAType: cython.int,
                       optInSignalPeriod: cython.int, optInSignalMAType: cython.int) -> cython.Py_ssize_t:
    """TA_MACDEXT_Lookback(optInFastPeriod, optInFastMAType, optInSlowPeriod, optInSlowMAType, optInSignalPeriod, optInSignalMAType) -> Py_ssize_t

    MACDEXT Lookback
    """
    if optInFastPeriod == 0:
        optInFastPeriod = 12
    if optInSlowPeriod == 0:
        optInSlowPeriod = 26
    if optInSignalPeriod == 0:
        optInSignalPeriod = 9

    if optInFastMAType == 0:
        optInFastMAType = 0
    if optInSlowMAType == 0:
        optInSlowMAType = 0
    if optInSignalMAType == 0:
        optInSignalMAType = 0

    if optInSlowPeriod < optInFastPeriod:
        tempInteger = optInSlowPeriod
        optInSlowPeriod = optInFastPeriod
        optInFastPeriod = tempInteger
        tempMAType = optInSlowMAType
        optInSlowMAType = optInFastMAType
        optInFastMAType = tempMAType

    lookbackLargest = TA_MA_Lookback(optInFastPeriod, optInFastMAType)
    tempInteger = TA_MA_Lookback(optInSlowPeriod, optInSlowMAType)
    if tempInteger > lookbackLargest:
        lookbackLargest = tempInteger

    return lookbackLargest + TA_MA_Lookback(optInSignalPeriod, optInSignalMAType)

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MACDEXT(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInFastPeriod: cython.int,
    optInFastMAType: cython.int,
    optInSlowPeriod: cython.int,
    optInSlowMAType: cython.int,
    optInSignalPeriod: cython.int,
    optInSignalMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMACD: cython.double[::1],
    outMACDSignal: cython.double[::1],
    outMACDHist: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_OUT_OF_RANGE_END_INDEX

    if optInFastPeriod == 0:
        optInFastPeriod = 12
    elif optInFastPeriod < 2 or optInFastPeriod > 100000:
        return TA_BAD_PARAM

    if optInSlowPeriod == 0:
        optInSlowPeriod = 26
    elif optInSlowPeriod < 2 or optInSlowPeriod > 100000:
        return TA_BAD_PARAM

    if optInSignalPeriod == 0:
        optInSignalPeriod = 9
    elif optInSignalPeriod < 1 or optInSignalPeriod > 100000:
        return TA_BAD_PARAM

    if optInFastMAType == 0:
        optInFastMAType = 0
    elif optInFastMAType < 0 or optInFastMAType > 8:
        return TA_BAD_PARAM

    if optInSlowMAType == 0:
        optInSlowMAType = 0
    elif optInSlowMAType < 0 or optInSlowMAType > 8:
        return TA_BAD_PARAM

    if optInSignalMAType == 0:
        optInSignalMAType = 0
    elif optInSignalMAType < 0 or optInSignalMAType > 8:
        return TA_BAD_PARAM

    # Make sure slow is really slower than the fast period
    if optInSlowPeriod < optInFastPeriod:
        tempInteger = optInSlowPeriod
        optInSlowPeriod = optInFastPeriod
        optInFastPeriod = tempInteger
        tempMAType = optInSlowMAType
        optInSlowMAType = optInFastMAType
        optInFastMAType = tempMAType

    lookbackLargest = TA_MA_Lookback(optInFastPeriod, optInFastMAType)
    tempInteger = TA_MA_Lookback(optInSlowPeriod, optInSlowMAType)
    if tempInteger > lookbackLargest:
        lookbackLargest = tempInteger

    lookbackSignal = TA_MA_Lookback(optInSignalPeriod, optInSignalMAType)
    lookbackTotal = lookbackSignal + lookbackLargest

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_SUCCESS

    tempInteger = (endIdx - startIdx) + 1 + lookbackSignal
    fastMABuffer: cython.double[::1] = np.zeros(tempInteger, dtype=np.float64)
    slowMABuffer: cython.double[::1] = np.zeros(tempInteger, dtype=np.float64)

    tempInteger = startIdx - lookbackSignal
    outBegIdx1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNbElement1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outBegIdx2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNbElement2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    # Calculate slow MA
    retCode = TA_MA(tempInteger, endIdx, inReal, optInSlowPeriod, optInSlowMAType,
                   outBegIdx1, outNbElement1, slowMABuffer)
    if retCode != TA_SUCCESS:
        return retCode

    # Calculate fast MA
    retCode = TA_MA(tempInteger, endIdx, inReal, optInFastPeriod, optInFastMAType,
                   outBegIdx2, outNbElement2, fastMABuffer)
    if retCode != TA_SUCCESS:
        return retCode

    # Calculate MACD line
    for i in range(outNbElement1[0]):
        fastMABuffer[i] = fastMABuffer[i] - slowMABuffer[i]

    # Copy MACD line to output
    outMACD[:outNbElement1[0] - lookbackSignal] = fastMABuffer[lookbackSignal:outNbElement1[0]]

    # Calculate signal line
    retCode = TA_MA(0, outNbElement1[0] - 1, fastMABuffer, optInSignalPeriod, optInSignalMAType,
                   outBegIdx2, outNbElement2, outMACDSignal)
    if retCode != TA_SUCCESS:
        return retCode

    # Calculate histogram
    for i in range(outNbElement2[0]):
        outMACDHist[i] = outMACD[i] - outMACDSignal[i]

    outBegIdx[0] = startIdx
    outNBElement[0] = outNbElement2[0]

    return TA_SUCCESS

def MACDEXT(real: np.ndarray, fastperiod: int = 12, fastmatype: int = 0,
           slowperiod: int = 26, slowmatype: int = 0,
           signalperiod: int = 9, signalmatype: int = 0):
    """MACDEXT(real, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    Moving Average Convergence/Divergence with controllable MA type

    Inputs:
        real: (any ndarray)
        fastperiod: (int) Number of period for the fast MA
        fastmatype: (int) Type of Moving Average for fast MA
        slowperiod: (int) Number of period for the slow MA
        slowmatype: (int) Type of Moving Average for slow MA
        signalperiod: (int) Smoothing for the signal line (nb of period)
        signalmatype: (int) Type of Moving Average for signal line
    Outputs:
        macd: (ndarray) MACD line
        macdsignal: (ndarray) Signal line
        macdhist: (ndarray) Histogram
    """
    real = check_array(real)

    outMACD = np.full_like(real, np.nan)
    outMACDSignal = np.full_like(real, np.nan)
    outMACDHist = np.full_like(real, np.nan)
    length = real.shape[0]

    startIdx = check_begidx1(real)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_MACDEXT_Lookback(fastperiod, fastmatype, slowperiod, slowmatype,
                                            signalperiod, signalmatype)

    outBegIdx = np.zeros(1, dtype=np.int64)
    outNBElement = np.zeros(1, dtype=np.int64)

    TA_MACDEXT(0, endIdx, real[startIdx:], fastperiod, fastmatype, slowperiod, slowmatype,
              signalperiod, signalmatype, outBegIdx, outNBElement,
              outMACD[lookback:], outMACDSignal[lookback:], outMACDHist[lookback:])
    return outMACD, outMACDSignal, outMACDHist 
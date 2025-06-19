import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from .ta_EMA import TA_EMA_Lookback, TA_INT_EMA
from .ta_MACD import TA_MACD_Lookback, TA_INT_MACD
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_MACDFIX_Lookback(optInSignalPeriod: cython.int) -> cython.Py_ssize_t:
    """TA_MACDFIX_Lookback(optInSignalPeriod) -> Py_ssize_t

    MACD Fix Lookback
    The lookback is driven by the signal line output.
    (must also account for the initial data consume by the fix 26 period EMA).

    Input:
        optInSignalPeriod: (int) Smoothing for the signal line (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInSignalPeriod == TA_INTEGER_DEFAULT:
            optInSignalPeriod = 9
        elif optInSignalPeriod < 1 or optInSignalPeriod > 100000:
            return -1
    # Compute the lookback period of the 26-period EMA and the signal period EMA
    return TA_EMA_Lookback(26) + TA_EMA_Lookback(optInSignalPeriod)

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MACDFIX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInSignalPeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMACD: cython.double[::1],
    outMACDSignal: cython.double[::1],
    outMACDHist: cython.double[::1],
) -> cython.int:
    """TA_MACDFIX - Moving Average Convergence/Divergence Fix 12/26

    Input  = double
    Output = double, double, double

    Optional Parameters
    -------------------
    optInSignalPeriod: (From 1 to 100000)
        Smoothing for the signal line (nb of period)
    """
    # Parameters check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal is None or outMACD is None or outMACDSignal is None or outMACDHist is None:
            return TA_RetCode.TA_BAD_PARAM
        # Check the signal period parameter
        if optInSignalPeriod == TA_INTEGER_DEFAULT:
            optInSignalPeriod = 9
        elif optInSignalPeriod < 1 or optInSignalPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

    retCode = TA_INT_MACD(
        startIdx,
        endIdx,
        inReal,
        0,
        0,
        optInSignalPeriod,
        outBegIdx,
        outNBElement,
        outMACD,
        outMACDSignal,
        outMACDHist
    )
    
    return retCode

def MACDFIX(real: np.ndarray, signalperiod: int = 9):
    """MACD(real, signalperiod=9)

    Moving Average Convergence/Divergence Fix 12/26

    Inputs:
        real: (any ndarray) Input array
        signalperiod: (int) Smoothing for the signal line (nb of period), default is 9

    Outputs:
        macd: (ndarray) MACD line
        macdsignal: (ndarray) Signal line
        macdhist: (ndarray) Histogram
    """
    real = check_array(real)

    outMACD = np.full_like(real, np.nan)
    outMACDSignal = np.full_like(real, np.nan)
    outMACDHist = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_MACDFIX_Lookback(signalperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_MACDFIX(
        0, 
        endIdx, 
        real[startIdx:], 
        signalperiod,
        outBegIdx, 
        outNBElement, 
        outMACD[lookback:], 
        outMACDSignal[lookback:], 
        outMACDHist[lookback:]
    )
    return outMACD, outMACDSignal, outMACDHist

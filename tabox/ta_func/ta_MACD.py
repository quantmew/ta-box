import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from .ta_EMA import TA_EMA, TA_EMA_Lookback, TA_INT_EMA
from .ta_utility import PER_TO_K
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

def TA_MACD_Lookback(
    optInFastPeriod: cython.int,
    optInSlowPeriod: cython.int,
    optInSignalPeriod: cython.int,
) -> cython.Py_ssize_t:
    """TA_MACD_Lookback(optInFastPeriod, optInSlowPeriod, optInSignalPeriod) -> Py_ssize_t

    MACD Lookback
    """
    if optInFastPeriod == TA_INTEGER_DEFAULT:
        optInFastPeriod = 12
    if optInSlowPeriod == TA_INTEGER_DEFAULT:
        optInSlowPeriod = 26
    if optInSignalPeriod == TA_INTEGER_DEFAULT:
        optInSignalPeriod = 9

    if optInSlowPeriod < optInFastPeriod:
        tempInteger = optInSlowPeriod
        optInSlowPeriod = optInFastPeriod
        optInFastPeriod = tempInteger

    return TA_EMA_Lookback(optInSlowPeriod) + TA_EMA_Lookback(optInSignalPeriod)


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_MACD(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInFastPeriod: cython.int,
    optInSlowPeriod: cython.int,
    optInSignalPeriod_2: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMACD: cython.double[::1],
    outMACDSignal: cython.double[::1],
    outMACDHist: cython.double[::1],
) -> cython.int:
    # Make sure slow is really slower than the fast period
    if optInSlowPeriod < optInFastPeriod:
        tempInteger = optInSlowPeriod
        optInSlowPeriod = optInFastPeriod
        optInFastPeriod = tempInteger

    k1: cython.double = 0.0
    k2: cython.double = 0.0

    if optInSlowPeriod != 0:
        k1 = PER_TO_K(optInSlowPeriod)
    else:
        optInSlowPeriod = 26
        k1 = 0.075  # fix 26

    if optInFastPeriod != 0:
        k2 = PER_TO_K(optInFastPeriod)
    else:
        optInFastPeriod = 12
        k2 = 0.15  # fix 12

    lookbackSignal = TA_EMA_Lookback(optInSignalPeriod_2)
    lookbackTotal = lookbackSignal + TA_EMA_Lookback(optInSlowPeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    tempInteger: cython.Py_ssize_t = (endIdx - startIdx) + 1 + lookbackSignal
    fastEMABuffer: cython.double[::1] = np.zeros(tempInteger, dtype=np.float64)
    slowEMABuffer: cython.double[::1] = np.zeros(tempInteger, dtype=np.float64)

    tempInteger: cython.Py_ssize_t = startIdx - lookbackSignal
    outBegIdx1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNbElement1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outBegIdx2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNbElement2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    # Calculate slow EMA
    retCode = TA_INT_EMA(
        tempInteger,
        endIdx,
        inReal,
        optInSlowPeriod,
        k1,
        outBegIdx1,
        outNbElement1,
        slowEMABuffer,
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    # Calculate fast EMA
    retCode = TA_INT_EMA(
        tempInteger,
        endIdx,
        inReal,
        optInFastPeriod,
        k2,
        outBegIdx2,
        outNbElement2,
        fastEMABuffer,
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    # Parano tests. Will be removed eventually.
    if (
        outBegIdx1[0] != tempInteger
        or outBegIdx2[0] != tempInteger
        or outNbElement1[0] != outNbElement2[0]
        or outNbElement1[0] != (endIdx - startIdx) + 1 + lookbackSignal
    ):
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_INTERNAL_ERROR

    # Calculate MACD line
    for i in range(outNbElement1[0]):
        fastEMABuffer[i] = fastEMABuffer[i] - slowEMABuffer[i]

    # Copy MACD line to output
    # for i in range(outNbElement1[0] - lookbackSignal):
    #     outMACD[i] = fastEMABuffer[i + lookbackSignal]
    outMACD[: outNbElement1[0] - lookbackSignal] = fastEMABuffer[
        lookbackSignal : outNbElement1[0]
    ]

    # Calculate signal line
    retCode = TA_INT_EMA(
        0,
        outNbElement1[0] - 1,
        fastEMABuffer,
        optInSignalPeriod_2,
        PER_TO_K(optInSignalPeriod_2),
        outBegIdx2,
        outNbElement2,
        outMACDSignal,
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    # Calculate histogram
    for i in range(outNbElement2[0]):
        outMACDHist[i] = outMACD[i] - outMACDSignal[i]

    outBegIdx[0] = startIdx
    outNBElement[0] = outNbElement2[0]
    return TA_RetCode.TA_SUCCESS


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MACD(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInFastPeriod: cython.int,
    optInSlowPeriod: cython.int,
    optInSignalPeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMACD: cython.double[::1],
    outMACDSignal: cython.double[::1],
    outMACDHist: cython.double[::1],
) -> cython.int:
    # Parameters check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if optInFastPeriod == 0:
            optInFastPeriod = 12
        elif optInFastPeriod < 2 or optInFastPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowPeriod == 0:
            optInSlowPeriod = 26
        elif optInSlowPeriod < 2 or optInSlowPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInSignalPeriod == 0:
            optInSignalPeriod = 9
        elif optInSignalPeriod < 1 or optInSignalPeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

    retCode = TA_INT_MACD(
        startIdx,
        endIdx,
        inReal,
        optInFastPeriod,
        optInSlowPeriod,
        optInSignalPeriod,
        outBegIdx,
        outNBElement,
        outMACD,
        outMACDSignal,
        outMACDHist,
    )
    return retCode


def MACD(
    real: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
):
    """MACD(real, fastperiod=12, slowperiod=26, signalperiod=9)

    Moving Average Convergence/Divergence

    Inputs:
        real: (any ndarray)
        fastperiod: (int) Number of period for the fast MA
        slowperiod: (int) Number of period for the slow MA
        signalperiod: (int) Smoothing for the signal line (nb of period)
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
    lookback = startIdx + TA_MACD_Lookback(fastperiod, slowperiod, signalperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_MACD(
        0,
        endIdx,
        real[startIdx:],
        fastperiod,
        slowperiod,
        signalperiod,
        outBegIdx,
        outNBElement,
        outMACD[lookback:],
        outMACDSignal[lookback:],
        outMACDHist[lookback:],
    )
    return outMACD, outMACDSignal, outMACDHist

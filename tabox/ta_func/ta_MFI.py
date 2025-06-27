import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

class MoneyFlow:
    def __init__(self):
        self.positive: cython.double = 0.0
        self.negative: cython.double = 0.0


def TA_MFI_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MFI_Lookback - Money Flow Index Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    return optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MFI)


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MFI(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    inVolume: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MFI - Money Flow Index

    Input  = High, Low, Close, Volume
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None or inVolume is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    posSumMF: cython.double
    negSumMF: cython.double
    prevValue: cython.double
    tempValue1: cython.double
    tempValue2: cython.double
    lookbackTotal: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t
    i: cython.Py_ssize_t
    today: cython.Py_ssize_t
    mflow_idx: cython.Py_ssize_t = 0
    mflow_size: cython.Py_ssize_t = optInTimePeriod
    mflow: list[MoneyFlow] = [MoneyFlow() for _ in range(mflow_size)]

    # Adjust startIdx to account for the lookback period
    lookbackTotal = optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(
        TA_FuncUnstId.TA_FUNC_UNST_MFI
    )

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx = 0  # Index into the output

    # Accumulate the positive and negative money flow among the initial period
    today = startIdx - lookbackTotal
    prevValue = (inHigh[today] + inLow[today] + inClose[today]) / 3.0

    posSumMF = 0.0
    negSumMF = 0.0
    today += 1
    for i in range(optInTimePeriod, 0, -1):
        tempValue1 = (inHigh[today] + inLow[today] + inClose[today]) / 3.0
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1
        tempValue1 *= inVolume[today]
        today += 1

        if tempValue2 < 0:
            mflow[mflow_idx].negative = tempValue1
            negSumMF += tempValue1
            mflow[mflow_idx].positive = 0.0
        elif tempValue2 > 0:
            mflow[mflow_idx].positive = tempValue1
            posSumMF += tempValue1
            mflow[mflow_idx].negative = 0.0
        else:
            mflow[mflow_idx].positive = 0.0
            mflow[mflow_idx].negative = 0.0

        mflow_idx = (mflow_idx + 1) % mflow_size

    # The following two equations are equivalent:
    #    MFI = 100 - (100 / 1 + (posSumMF/negSumMF))
    #    MFI = 100 * (posSumMF/(posSumMF+negSumMF))
    # The second equation is used here for speed optimization
    if today > startIdx:
        tempValue1 = posSumMF + negSumMF
        if tempValue1 < 1.0:
            outReal[outIdx] = 0.0
        else:
            outReal[outIdx] = 100.0 * (posSumMF / tempValue1)
        outIdx += 1
    else:
        # Skip the unstable period. Do the processing but do not write it in the output
        while today < startIdx:
            posSumMF -= mflow[mflow_idx].positive
            negSumMF -= mflow[mflow_idx].negative

            tempValue1 = (inHigh[today] + inLow[today] + inClose[today]) / 3.0
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1
            tempValue1 *= inVolume[today]
            today += 1

            if tempValue2 < 0:
                mflow[mflow_idx].negative = tempValue1
                negSumMF += tempValue1
                mflow[mflow_idx].positive = 0.0
            elif tempValue2 > 0:
                mflow[mflow_idx].positive = tempValue1
                posSumMF += tempValue1
                mflow[mflow_idx].negative = 0.0
            else:
                mflow[mflow_idx].positive = 0.0
                mflow[mflow_idx].negative = 0.0
            
            mflow_idx = (mflow_idx + 1) % mflow_size

    # Unstable period skipped... now continue processing if needed
    while today <= endIdx:
        posSumMF -= mflow[mflow_idx].positive
        negSumMF -= mflow[mflow_idx].negative

        tempValue1 = (inHigh[today] + inLow[today] + inClose[today]) / 3.0
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1
        tempValue1 *= inVolume[today]
        today += 1

        if tempValue2 < 0:
            mflow[mflow_idx].negative = tempValue1
            negSumMF += tempValue1
            mflow[mflow_idx].positive = 0.0
        elif tempValue2 > 0:
            mflow[mflow_idx].positive = tempValue1
            posSumMF += tempValue1
            mflow[mflow_idx].negative = 0.0
        else:
            mflow[mflow_idx].positive = 0.0
            mflow[mflow_idx].negative = 0.0

        tempValue1 = posSumMF + negSumMF
        if tempValue1 < 1.0:
            outReal[outIdx] = 0.0
        else:
            outReal[outIdx] = 100.0 * (posSumMF / tempValue1)
        outIdx += 1
        mflow_idx = (mflow_idx + 1) % mflow_size

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def MFI(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """MFI(high, low, close, volume[, timeperiod=14])

    Money Flow Index (Volume Indicators)

    The Money Flow Index (MFI) is a momentum indicator that uses both price and volume
    to identify overbought or oversold conditions in a market.

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
        close: (any ndarray) Close prices
        volume: (any ndarray) Volume
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)

    if (
        high.shape != low.shape
        or high.shape != close.shape
        or high.shape != volume.shape
    ):
        raise ValueError("Input arrays must have the same shape")

    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MFI_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MFI(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        volume[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

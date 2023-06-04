
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod, make_double_array

def TA_RSI_Lookback(optInTimePeriod: cython.int) -> cython.int:
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_RSI(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int,
           outBegIdx: cython.int[::1], outNBElement: cython.int[::1], outReal: cython.double[::1]) -> None:
    # Adjust startIdx to account for the lookback period.
    lookbackTotal = TA_RSI_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        return 0

    outIdx = 0 # Index into the output.

    """
    Trap special case where the period is '1'.
    In that case, just copy the input into the
    output for the requested range (as-is !)
    """
    if optInTimePeriod == 1:
        outBegIdx[0] = startIdx
        i = (endIdx - startIdx) + 1
        outNBElement[0] = i
        outReal[:i] = inReal[startIdx:startIdx+i]
        return 0

    # Initialize the up/down sums and averages.
    up_sum = down_sum = up_avg = down_avg = 0.0

    # Calculate the initial up/down sums and averages.
    for i in range(startIdx - lookbackTotal, startIdx):
        diff = inReal[i] - inReal[i - 1]
        if diff > 0:
            up_sum += diff
        else:
            down_sum -= diff

    up_avg = up_sum / optInTimePeriod
    down_avg = down_sum / optInTimePeriod

    # Calculate the RSI for the remaining data.
    outBegIdx[0] = startIdx
    outIdx = 0
    for i in range(startIdx, endIdx + 1):
        diff = inReal[i] - inReal[i - 1]
        if diff > 0:
            up_avg = ((optInTimePeriod - 1) * up_avg + diff) / optInTimePeriod
            down_avg = ((optInTimePeriod - 1) * down_avg) / optInTimePeriod
        else:
            up_avg = ((optInTimePeriod - 1) * up_avg) / optInTimePeriod
            down_avg = ((optInTimePeriod - 1) * down_avg - diff) / optInTimePeriod

        rs = up_avg / down_avg
        rsi = 100.0 - (100.0 / (1.0 + rs))

        outReal[outIdx] = rsi
        outIdx += 1

    outNBElement[0] = outIdx

    return outIdx


def RSI(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """ RSI(real[, timeperiod=?])

    Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)
    
    length: cython.int = real.shape[0]
    begidx: cython.int = check_begidx1(real)
    endidx: cython.int = length - begidx - 1
    lookback = begidx + TA_RSI_Lookback(timeperiod)
    outReal = make_double_array(length, lookback)
    outBegIdx: cython.int[::1] = np.zeros(shape=(1,), dtype=np.int32)
    outNBElement: cython.int[::1] = np.zeros(shape=(1,), dtype=np.int32)

    retCode = TA_RSI(0, endidx, real[begidx:], timeperiod, outBegIdx, outNBElement, outReal[lookback:])
    return outReal 
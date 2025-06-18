import cython
import numpy as np
from tabox.settings import TA_FUNC_NO_RANGE_CHECK

from tabox.ta_func.ta_utility import TA_IS_ZERO
from .ta_utils import check_array, check_begidx1, check_timeperiod, make_double_array
from ..retcode import TA_RetCode


def TA_RSI_Lookback(optInTimePeriod: cython.int) -> cython.int:
    return optInTimePeriod


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_RSI(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    outIdx: cython.Py_ssize_t = 0

    today: cython.Py_ssize_t = 0
    lookbackTotal: cython.Py_ssize_t = 0
    unstablePeriod: cython.bint = False
    i: cython.Py_ssize_t = 0

    prevGain: cython.double = 0.0
    prevLoss: cython.double = 0.0
    prevValue: cython.double = 0.0
    savePrevValue: cython.double = 0.0
    tempValue1: cython.double = 0.0
    tempValue2: cython.double = 0.0

    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        # min/max are checked for optInTimePeriod.
        if optInTimePeriod == TA_RetCode.TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif (optInTimePeriod < 2) or (optInTimePeriod > 100000):
            return TA_RetCode.TA_BAD_PARAM

    outBegIdx[0] = 0
    outNBElement[0] = 0
    # Adjust startIdx to account for the lookback period.
    lookbackTotal = TA_RSI_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    outIdx = 0  # Index into the output.

    """
    Trap special case where the period is '1'.
    In that case, just copy the input into the
    output for the requested range (as-is !)
    """
    if optInTimePeriod == 1:
        outBegIdx[0] = startIdx
        i = (endIdx - startIdx) + 1
        outNBElement[0] = i
        outReal[:i] = inReal[startIdx : startIdx + i]
        return TA_RetCode.TA_SUCCESS

    """
    Accumulate Wilder's "Average Gain" and "Average Loss" 
    among the initial period.
    """

    today = startIdx - lookbackTotal
    prevValue = inReal[today]

    unstablePeriod = True
    if not unstablePeriod:
        """
        Preserve prevValue because it may get
        overwritten by the output.
        (because output ptr could be the same as input ptr).
        """
        savePrevValue = prevValue

        """
        No unstable period, so must calculate first output
        particular to Metastock.
        (Metastock re-use the first price bar, so there
        is no loss/gain at first. Beats me why they
        are doing all this).
        """
        prevGain = 0.0
        prevLoss = 0.0

        i: cython.Py_ssize_t = optInTimePeriod
        while i > 0:
            tempValue1 = inReal[today]
            today += 1
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1
            if tempValue2 < 0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2
            i -= 1

        tempValue1 = prevLoss / optInTimePeriod
        tempValue2 = prevGain / optInTimePeriod

        # Write the output.
        tempValue1 = tempValue2 + tempValue1
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100 * (tempValue2 / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1

        # Are we done?
        if today > endIdx:
            outBegIdx[0] = startIdx
            outNBElement[0] = outIdx
            return TA_RetCode.TA_SUCCESS

        # Start over for the next price bar.
        today -= optInTimePeriod
        prevValue = savePrevValue

    """
    Remaining of the processing is identical
    for both Classic calculation and Metastock.
    """
    prevGain = 0.0
    prevLoss = 0.0
    today += 1

    i: cython.Py_ssize_t = optInTimePeriod
    while i > 0:
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1
        if tempValue2 < 0.0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2
        i -= 1

    """
    Subsequent prevLoss and prevGain are smoothed
    using the previous values (Wilder's approach).
    1) Multiply the previous by 'period-1'. 
    2) Add today value.
    3) Divide by 'period'.
    """
    prevLoss /= optInTimePeriod
    prevGain /= optInTimePeriod

    """
    Often documentation present the RSI calculation as follow:
        RSI = 100 - (100 / 1 + (prevGain/prevLoss))
    
    The following is equivalent:
        RSI = 100 * (prevGain/(prevGain+prevLoss))
    
    The second equation is used here for speed optimization.
    """
    if today > startIdx:
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * (prevGain / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1
    else:
        """
        Skip the unstable period. Do the processing
        but do not write it in the output.
        """
        while today < startIdx:
            tempValue1 = inReal[today]
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1

            prevLoss *= optInTimePeriod - 1
            prevGain *= optInTimePeriod - 1
            if tempValue2 < 0.0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2

            prevLoss /= optInTimePeriod
            prevGain /= optInTimePeriod

            today += 1

    """
    Unstable period skipped... now continue
    processing if needed.
    """
    while today <= endIdx:
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1

        prevLoss *= optInTimePeriod - 1
        prevGain *= optInTimePeriod - 1
        if tempValue2 < 0.0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2

        prevLoss /= optInTimePeriod
        prevGain /= optInTimePeriod
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * (prevGain / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_S_RSI(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.float[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.float[::1],
) -> cython.int:
    outIdx: cython.Py_ssize_t = 0

    today: cython.Py_ssize_t = 0
    lookbackTotal: cython.Py_ssize_t = 0
    unstablePeriod: cython.bint = False
    i: cython.Py_ssize_t = 0

    prevGain: cython.float = 0.0
    prevLoss: cython.float = 0.0
    prevValue: cython.float = 0.0
    savePrevValue: cython.float = 0.0
    tempValue1: cython.float = 0.0
    tempValue2: cython.float = 0.0

    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        # min/max are checked for optInTimePeriod.
        if optInTimePeriod == TA_RetCode.TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif (optInTimePeriod < 2) or (optInTimePeriod > 100000):
            return TA_RetCode.TA_BAD_PARAM

    outBegIdx[0] = 0
    outNBElement[0] = 0
    # Adjust startIdx to account for the lookback period.
    lookbackTotal = TA_RSI_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    outIdx = 0  # Index into the output.

    """
    Trap special case where the period is '1'.
    In that case, just copy the input into the
    output for the requested range (as-is !)
    """
    if optInTimePeriod == 1:
        outBegIdx[0] = startIdx
        i = (endIdx - startIdx) + 1
        outNBElement[0] = i
        outReal[:i] = inReal[startIdx : startIdx + i]
        return TA_RetCode.TA_SUCCESS

    """
    Accumulate Wilder's "Average Gain" and "Average Loss" 
    among the initial period.
    """

    today = startIdx - lookbackTotal
    prevValue = inReal[today]

    unstablePeriod = True
    if not unstablePeriod:
        """
        Preserve prevValue because it may get
        overwritten by the output.
        (because output ptr could be the same as input ptr).
        """
        savePrevValue = prevValue

        """
        No unstable period, so must calculate first output
        particular to Metastock.
        (Metastock re-use the first price bar, so there
        is no loss/gain at first. Beats me why they
        are doing all this).
        """
        prevGain = 0.0
        prevLoss = 0.0

        i: cython.int = optInTimePeriod
        while i > 0:
            tempValue1 = inReal[today]
            today += 1
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1
            if tempValue2 < 0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2
            i -= 1

        tempValue1 = prevLoss / optInTimePeriod
        tempValue2 = prevGain / optInTimePeriod

        # Write the output.
        tempValue1 = tempValue2 + tempValue1
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100 * (tempValue2 / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1

        # Are we done?
        if today > endIdx:
            outBegIdx[0] = startIdx
            outNBElement[0] = outIdx
            return TA_RetCode.TA_SUCCESS

        # Start over for the next price bar.
        today -= optInTimePeriod
        prevValue = savePrevValue

    """
    Remaining of the processing is identical
    for both Classic calculation and Metastock.
    """
    prevGain = 0.0
    prevLoss = 0.0
    today += 1

    i: cython.int = optInTimePeriod
    while i > 0:
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1
        if tempValue2 < 0.0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2
        i -= 1

    """
    Subsequent prevLoss and prevGain are smoothed
    using the previous values (Wilder's approach).
    1) Multiply the previous by 'period-1'. 
    2) Add today value.
    3) Divide by 'period'.
    """
    prevLoss /= optInTimePeriod
    prevGain /= optInTimePeriod

    """
    Often documentation present the RSI calculation as follow:
        RSI = 100 - (100 / 1 + (prevGain/prevLoss))
    
    The following is equivalent:
        RSI = 100 * (prevGain/(prevGain+prevLoss))
    
    The second equation is used here for speed optimization.
    """
    if today > startIdx:
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * (prevGain / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1
    else:
        """
        Skip the unstable period. Do the processing
        but do not write it in the output.
        """
        while today < startIdx:
            tempValue1 = inReal[today]
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1

            prevLoss *= optInTimePeriod - 1
            prevGain *= optInTimePeriod - 1
            if tempValue2 < 0.0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2

            prevLoss /= optInTimePeriod
            prevGain /= optInTimePeriod

            today += 1

    """
    Unstable period skipped... now continue
    processing if needed.
    """
    while today <= endIdx:
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1

        prevLoss *= optInTimePeriod - 1
        prevGain *= optInTimePeriod - 1
        if tempValue2 < 0.0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2

        prevLoss /= optInTimePeriod
        prevGain /= optInTimePeriod
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * (prevGain / tempValue1)
            outIdx += 1
        else:
            outReal[outIdx] = 0.0
            outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


def RSI(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """RSI(real[, timeperiod=?])

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

    length: cython.Py_ssize_t = real.shape[0]
    begidx: cython.Py_ssize_t = check_begidx1(real)
    endidx: cython.Py_ssize_t = length - begidx - 1
    lookback = begidx + TA_RSI_Lookback(timeperiod)
    outReal = make_double_array(length, lookback)
    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(shape=(1,), dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(shape=(1,), dtype=np.int64)

    retCode = TA_RSI(
        0,
        endidx,
        real[begidx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    return outReal

import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_GLOBALS_COMPATIBILITY, TA_Compatibility, TA_FuncUnstId, TA_IS_ZERO
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

def TA_CMO_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_CMO_Lookback - Chande Momentum Oscillator Lookback

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

    unstable_period = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_CMO)
    ret_value = optInTimePeriod + unstable_period

    # Handle Metastock compatibility
    if TA_GLOBALS_COMPATIBILITY() == TA_Compatibility.TA_COMPATIBILITY_METASTOCK:
        ret_value -= 1

    return ret_value


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_CMO(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_CMO - Chande Momentum Oscillator

    Input  = double
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
        if optInTimePeriod == 0:  # 默认值处理
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    unstablePeriod: cython.Py_ssize_t
    i: cython.Py_ssize_t
    prevGain: cython.double
    prevLoss: cython.double
    prevValue: cython.double
    savePrevValue: cython.double
    tempValue1: cython.double
    tempValue2: cython.double
    tempValue3: cython.double
    tempValue4: cython.double

    # Adjust startIdx to account for the lookback period
    lookbackTotal = TA_CMO_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Trap special case where the period is '1'
    if optInTimePeriod == 1:
        outBegIdx[0] = startIdx
        i = endIdx - startIdx + 1
        outNBElement[0] = i
        # Copy input to output
        for idx in range(i):
            outReal[idx] = inReal[startIdx + idx]
        return TA_RetCode.TA_SUCCESS

    # Accumulate Wilder's "Average Gain" and "Average Loss" among the initial period
    today = startIdx - lookbackTotal
    prevValue = inReal[today]

    unstablePeriod = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_CMO)

    # Handle Metastock compatibility for initial calculation
    if (unstablePeriod == 0) and (TA_GLOBALS_COMPATIBILITY() == TA_Compatibility.TA_COMPATIBILITY_METASTOCK):
        savePrevValue = prevValue
        prevGain = 0.0
        prevLoss = 0.0
        for i in range(optInTimePeriod, 0, -1):
            tempValue1 = inReal[today]
            today += 1
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1
            if tempValue2 < 0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2

        tempValue1 = prevLoss / optInTimePeriod
        tempValue2 = prevGain / optInTimePeriod
        tempValue3 = tempValue2 - tempValue1
        tempValue4 = tempValue1 + tempValue2

        # Write the output
        if not TA_IS_ZERO(tempValue4):
            outReal[outIdx] = 100 * (tempValue3 / tempValue4)
        else:
            outReal[outIdx] = 0.0
        outIdx += 1

        # Check if done
        if today > endIdx:
            outBegIdx[0] = startIdx
            outNBElement[0] = outIdx
            return TA_RetCode.TA_SUCCESS

        # Reset for next calculation
        today -= optInTimePeriod
        prevValue = savePrevValue

    # Main calculation for CMO
    prevGain = 0.0
    prevLoss = 0.0
    today += 1
    for i in range(optInTimePeriod, 0, -1):
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1
        if tempValue2 < 0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2

    # Smooth the initial gains and losses
    prevLoss /= optInTimePeriod
    prevGain /= optInTimePeriod

    # Skip the unstable period if needed
    if today > startIdx:
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * ((prevGain - prevLoss) / tempValue1)
        else:
            outReal[outIdx] = 0.0
        outIdx += 1
    else:
        while today < startIdx:
            tempValue1 = inReal[today]
            today += 1
            tempValue2 = tempValue1 - prevValue
            prevValue = tempValue1

            prevLoss *= optInTimePeriod - 1
            prevGain *= optInTimePeriod - 1
            if tempValue2 < 0:
                prevLoss -= tempValue2
            else:
                prevGain += tempValue2

            prevLoss /= optInTimePeriod
            prevGain /= optInTimePeriod

    # Calculate remaining values
    while today <= endIdx:
        tempValue1 = inReal[today]
        today += 1
        tempValue2 = tempValue1 - prevValue
        prevValue = tempValue1

        prevLoss *= optInTimePeriod - 1
        prevGain *= optInTimePeriod - 1
        if tempValue2 < 0:
            prevLoss -= tempValue2
        else:
            prevGain += tempValue2

        prevLoss /= optInTimePeriod
        prevGain /= optInTimePeriod
        tempValue1 = prevGain + prevLoss
        if not TA_IS_ZERO(tempValue1):
            outReal[outIdx] = 100.0 * ((prevGain - prevLoss) / tempValue1)
        else:
            outReal[outIdx] = 0.0
        outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def CMO(real: np.ndarray, timeperiod: int = 14):
    """CMO(real[, timeperiod=14])

    Chande Momentum Oscillator (Overlap Studies)

    The CMO is calculated by taking the difference between the sum of gains and the sum of losses
    over a given period, divided by the sum of gains and losses, multiplied by 100.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_CMO_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_CMO(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

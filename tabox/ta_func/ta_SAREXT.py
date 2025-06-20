import cython
import numpy as np
from .ta_utils import check_array, check_begidx2
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MINUS_DM import TA_MINUS_DM


def TA_SAREXT_Lookback(
    optInStartValue: cython.double,
    optInOffsetOnReverse: cython.double,
    optInAccelerationInitLong: cython.double,
    optInAccelerationLong: cython.double,
    optInAccelerationMaxLong: cython.double,
    optInAccelerationInitShort: cython.double,
    optInAccelerationShort: cython.double,
    optInAccelerationMaxShort: cython.double,
) -> cython.Py_ssize_t:
    """
    TA_SAREXT_Lookback - Parabolic SAR - Extended Lookback

    Input:
        optInStartValue: (float) Start value and direction. 0 for Auto, >0 for Long, <0 for Short
        optInOffsetOnReverse: (float) Percent offset added/removed to initial stop on short/long reversal
        optInAccelerationInitLong: (float) Acceleration Factor initial value for the Long direction
        optInAccelerationLong: (float) Acceleration Factor for the Long direction
        optInAccelerationMaxLong: (float) Acceleration Factor maximum value for the Long direction
        optInAccelerationInitShort: (float) Acceleration Factor initial value for the Short direction
        optInAccelerationShort: (float) Acceleration Factor for the Short direction
        optInAccelerationMaxShort: (float) Acceleration Factor maximum value for the Short direction

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInStartValue == 0.0:
            optInStartValue = 0.0
        elif optInStartValue < -3.0e37 or optInStartValue > 3.0e37:
            return -1

        if optInOffsetOnReverse == 0.0:
            optInOffsetOnReverse = 0.0
        elif optInOffsetOnReverse < 0.0 or optInOffsetOnReverse > 3.0e37:
            return -1

        if optInAccelerationInitLong == 0.0:
            optInAccelerationInitLong = 0.02
        elif optInAccelerationInitLong < 0.0 or optInAccelerationInitLong > 3.0e37:
            return -1

        if optInAccelerationLong == 0.0:
            optInAccelerationLong = 0.02
        elif optInAccelerationLong < 0.0 or optInAccelerationLong > 3.0e37:
            return -1

        if optInAccelerationMaxLong == 0.0:
            optInAccelerationMaxLong = 0.2
        elif optInAccelerationMaxLong < 0.0 or optInAccelerationMaxLong > 3.0e37:
            return -1

        if optInAccelerationInitShort == 0.0:
            optInAccelerationInitShort = 0.02
        elif optInAccelerationInitShort < 0.0 or optInAccelerationInitShort > 3.0e37:
            return -1

        if optInAccelerationShort == 0.0:
            optInAccelerationShort = 0.02
        elif optInAccelerationShort < 0.0 or optInAccelerationShort > 3.0e37:
            return -1

        if optInAccelerationMaxShort == 0.0:
            optInAccelerationMaxShort = 0.2
        elif optInAccelerationMaxShort < 0.0 or optInAccelerationMaxShort > 3.0e37:
            return -1

    # SAR always sacrifices one price bar to establish the initial extreme price
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_SAREXT(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInStartValue: cython.double,
    optInOffsetOnReverse: cython.double,
    optInAccelerationInitLong: cython.double,
    optInAccelerationLong: cython.double,
    optInAccelerationMaxLong: cython.double,
    optInAccelerationInitShort: cython.double,
    optInAccelerationShort: cython.double,
    optInAccelerationMaxShort: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_SAREXT - Parabolic SAR - Extended

    Input  = High, Low
    Output = double

    Optional Parameters
    -------------------
    optInStartValue:(From TA_REAL_MIN to TA_REAL_MAX)
       Start value and direction. 0 for Auto, >0 for Long, <0 for Short
    optInOffsetOnReverse:(From 0 to TA_REAL_MAX)
       Percent offset added/removed to initial stop on short/long reversal
    optInAccelerationInitLong:(From 0 to TA_REAL_MAX)
       Acceleration Factor initial value for the Long direction
    optInAccelerationLong:(From 0 to TA_REAL_MAX)
       Acceleration Factor for the Long direction
    optInAccelerationMaxLong:(From 0 to TA_REAL_MAX)
       Acceleration Factor maximum value for the Long direction
    optInAccelerationInitShort:(From 0 to TA_REAL_MAX)
       Acceleration Factor initial value for the Short direction
    optInAccelerationShort:(From 0 to TA_REAL_MAX)
       Acceleration Factor for the Short direction
    optInAccelerationMaxShort:(From 0 to TA_REAL_MAX)
       Acceleration Factor maximum value for the Short direction
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if optInStartValue == 0.0:
            optInStartValue = 0.0
        elif optInStartValue < -3.0e37 or optInStartValue > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInOffsetOnReverse == 0.0:
            optInOffsetOnReverse = 0.0
        elif optInOffsetOnReverse < 0.0 or optInOffsetOnReverse > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationInitLong == 0.0:
            optInAccelerationInitLong = 0.02
        elif optInAccelerationInitLong < 0.0 or optInAccelerationInitLong > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationLong == 0.0:
            optInAccelerationLong = 0.02
        elif optInAccelerationLong < 0.0 or optInAccelerationLong > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationMaxLong == 0.0:
            optInAccelerationMaxLong = 0.2
        elif optInAccelerationMaxLong < 0.0 or optInAccelerationMaxLong > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationInitShort == 0.0:
            optInAccelerationInitShort = 0.02
        elif optInAccelerationInitShort < 0.0 or optInAccelerationInitShort > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationShort == 0.0:
            optInAccelerationShort = 0.02
        elif optInAccelerationShort < 0.0 or optInAccelerationShort > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if optInAccelerationMaxShort == 0.0:
            optInAccelerationMaxShort = 0.2
        elif optInAccelerationMaxShort < 0.0 or optInAccelerationMaxShort > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM

        if inHigh is None or inLow is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Move up the start index if there is not enough initial data
    if startIdx < 1:
        startIdx = 1

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Initialize acceleration factors
    afLong: cython.double = optInAccelerationInitLong
    afShort: cython.double = optInAccelerationInitShort

    # Make sure the acceleration and maximum are coherent
    if afLong > optInAccelerationMaxLong:
        afLong = optInAccelerationInitLong = optInAccelerationMaxLong
    if optInAccelerationLong > optInAccelerationMaxLong:
        optInAccelerationLong = optInAccelerationMaxLong
    if afShort > optInAccelerationMaxShort:
        afShort = optInAccelerationInitShort = optInAccelerationMaxShort
    if optInAccelerationShort > optInAccelerationMaxShort:
        optInAccelerationShort = optInAccelerationMaxShort

    # Initialize SAR calculations
    isLong: cython.int
    if optInStartValue == 0.0:  # Default action
        # Identify if the initial direction is long or short
        minus_dm_out = np.zeros(1, dtype=np.float64)
        minus_dm_beg = np.zeros(1, dtype=np.intp)
        minus_dm_nbe = np.zeros(1, dtype=np.intp)

        retCode = TA_MINUS_DM(
            startIdx,
            startIdx,
            inHigh,
            inLow,
            1,
            minus_dm_beg,
            minus_dm_nbe,
            minus_dm_out,
        )

        if retCode != TA_RetCode.TA_SUCCESS:
            outBegIdx[0] = 0
            outNBElement[0] = 0
            return retCode

        if minus_dm_out[0] > 0:
            isLong = 0
        else:
            isLong = 1
    elif optInStartValue > 0:  # Start Long
        isLong = 1
    else:  # optInStartValue < 0 => Start Short
        isLong = 0

    outBegIdx[0] = startIdx
    outIdx: cython.Py_ssize_t = 0

    # Write the first SAR
    todayIdx: cython.Py_ssize_t = startIdx
    newHigh: cython.double = inHigh[todayIdx - 1]
    newLow: cython.double = inLow[todayIdx - 1]

    ep: cython.double
    sar: cython.double

    if optInStartValue == 0.0:  # Default action
        if isLong == 1:
            ep = inHigh[todayIdx]
            sar = newLow
        else:
            ep = inLow[todayIdx]
            sar = newHigh
    elif optInStartValue > 0:  # Start Long at specified value
        ep = inHigh[todayIdx]
        sar = optInStartValue
    else:  # optInStartValue < 0 => Start Short at specified value
        ep = inLow[todayIdx]
        sar = abs(optInStartValue)

    # Cheat on the newLow and newHigh for the first iteration
    newLow = inLow[todayIdx]
    newHigh = inHigh[todayIdx]

    prevHigh: cython.double
    prevLow: cython.double

    while todayIdx <= endIdx:
        prevLow = newLow
        prevHigh = newHigh
        newLow = inLow[todayIdx]
        newHigh = inHigh[todayIdx]
        todayIdx += 1

        if isLong == 1:
            # Switch to short if the low penetrates the SAR value
            if newLow <= sar:
                # Switch and Override the SAR with the ep
                isLong = 0
                sar = ep

                # Make sure the override SAR is within yesterday's and today's range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh

                # Output the override SAR
                if optInOffsetOnReverse != 0.0:
                    sar += sar * optInOffsetOnReverse
                outReal[outIdx] = -sar
                outIdx += 1

                # Adjust afShort and ep
                afShort = optInAccelerationInitShort
                ep = newLow

                # Calculate the new SAR
                sar = sar + afShort * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh
            else:
                # No switch
                # Output the SAR (was calculated in the previous iteration)
                outReal[outIdx] = sar
                outIdx += 1

                # Adjust afLong and ep
                if newHigh > ep:
                    ep = newHigh
                    afLong += optInAccelerationLong
                    if afLong > optInAccelerationMaxLong:
                        afLong = optInAccelerationMaxLong

                # Calculate the new SAR
                sar = sar + afLong * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow
        else:
            # Switch to long if the high penetrates the SAR value
            if newHigh >= sar:
                # Switch and Override the SAR with the ep
                isLong = 1
                sar = ep

                # Make sure the override SAR is within yesterday's and today's range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow

                # Output the override SAR
                if optInOffsetOnReverse != 0.0:
                    sar -= sar * optInOffsetOnReverse
                outReal[outIdx] = sar
                outIdx += 1

                # Adjust afLong and ep
                afLong = optInAccelerationInitLong
                ep = newHigh

                # Calculate the new SAR
                sar = sar + afLong * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow
            else:
                # No switch
                # Output the SAR (was calculated in the previous iteration)
                outReal[outIdx] = -sar
                outIdx += 1

                # Adjust afShort and ep
                if newLow < ep:
                    ep = newLow
                    afShort += optInAccelerationShort
                    if afShort > optInAccelerationMaxShort:
                        afShort = optInAccelerationMaxShort

                # Calculate the new SAR
                sar = sar + afShort * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def SAREXT(
    high: np.ndarray,
    low: np.ndarray,
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.02,
    accelerationlong: float = 0.02,
    accelerationmaxlong: float = 0.2,
    accelerationinitshort: float = 0.02,
    accelerationshort: float = 0.02,
    accelerationmaxshort: float = 0.2,
) -> np.ndarray:
    """SAREXT(high, low[, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02,
             accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02,
             accelerationshort=0.02, accelerationmaxshort=0.2])

    Parabolic SAR - Extended (Overlap Studies)

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
    Parameters:
        startvalue: 0 Start value and direction (0=auto, >0=long, <0=short)
        offsetonreverse: 0 Percent offset added/removed on reversal
        accelerationinitlong: 0.02 Acceleration Factor initial value for Long
        accelerationlong: 0.02 Acceleration Factor for Long
        accelerationmaxlong: 0.2 Acceleration Factor maximum value for Long
        accelerationinitshort: 0.02 Acceleration Factor initial value for Short
        accelerationshort: 0.02 Acceleration Factor for Short
        accelerationmaxshort: 0.2 Acceleration Factor maximum value for Short
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx2(high, low)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_SAREXT_Lookback(
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort,
    )

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_SAREXT(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal

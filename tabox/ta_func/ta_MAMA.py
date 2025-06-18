import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
import math

def TA_MAMA_Lookback(optInFastLimit: cython.double, optInSlowLimit: cython.double) -> cython.Py_ssize_t:
    """TA_MAMA_Lookback(optInFastLimit, optInSlowLimit) -> Py_ssize_t

    MAMA Lookback
    """
    if optInFastLimit == 0:
        optInFastLimit = 0.5
    elif optInFastLimit < 0.01 or optInFastLimit > 0.99:
        return -1

    if optInSlowLimit == 0:
        optInSlowLimit = 0.05
    elif optInSlowLimit < 0.01 or optInSlowLimit > 0.99:
        return -1

    return 32 + 2  # TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_MAMA,Mama)

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MAMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInFastLimit: cython.double,
    optInSlowLimit: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMAMA: cython.double[::1],
    outFAMA: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    if optInFastLimit == 0:
        optInFastLimit = 0.5
    elif optInFastLimit < 0.01 or optInFastLimit > 0.99:
        return TA_RetCode.TA_BAD_PARAM

    if optInSlowLimit == 0:
        optInSlowLimit = 0.05
    elif optInSlowLimit < 0.01 or optInSlowLimit > 0.99:
        return TA_RetCode.TA_BAD_PARAM

    rad2Deg = 180.0 / (4.0 * math.atan(1))
    lookbackTotal = 32 + 2  # TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_MAMA,Mama)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx

    trailingWMAIdx = startIdx - lookbackTotal
    today = trailingWMAIdx

    tempReal = inReal[today]
    today += 1
    periodWMASub = tempReal
    periodWMASum = tempReal
    tempReal = inReal[today]
    today += 1
    periodWMASub += tempReal
    periodWMASum += tempReal * 2.0
    tempReal = inReal[today]
    today += 1
    periodWMASub += tempReal
    periodWMASum += tempReal * 3.0

    trailingWMAValue = 0.0

    i = 9
    while i > 0:
        tempReal = inReal[today]
        today += 1
        periodWMASub += tempReal
        periodWMASub -= trailingWMAValue
        periodWMASum += tempReal * 4.0
        trailingWMAValue = inReal[trailingWMAIdx]
        trailingWMAIdx += 1
        smoothedValue = periodWMASum * 0.1
        periodWMASum -= periodWMASub
        i -= 1

    hilbertIdx = 0

    detrender = [0.0] * 3
    Q1 = [0.0] * 3
    jI = [0.0] * 3
    jQ = [0.0] * 3

    period = 0.0
    outIdx = 0

    prevI2 = prevQ2 = 0.0
    Re = Im = 0.0
    mama = fama = 0.0
    I1ForOddPrev3 = I1ForEvenPrev3 = 0.0
    I1ForOddPrev2 = I1ForEvenPrev2 = 0.0

    prevPhase = 0.0

    while today <= endIdx:
        adjustedPrevPeriod = (0.075 * period) + 0.54

        todayValue = inReal[today]
        periodWMASub += todayValue
        periodWMASub -= trailingWMAValue
        periodWMASum += todayValue * 4.0
        trailingWMAValue = inReal[trailingWMAIdx]
        trailingWMAIdx += 1
        smoothedValue = periodWMASum * 0.1
        periodWMASum -= periodWMASub

        if (today % 2) == 0:
            detrender[hilbertIdx] = (0.0962 * smoothedValue) + (0.5769 * detrender[1]) - (0.5769 * detrender[2]) - (0.0962 * detrender[0])
            Q1[hilbertIdx] = (0.0962 * detrender[hilbertIdx]) + (0.5769 * Q1[1]) - (0.5769 * Q1[2]) - (0.0962 * Q1[0])
            jI[hilbertIdx] = (0.0962 * I1ForEvenPrev3) + (0.5769 * jI[1]) - (0.5769 * jI[2]) - (0.0962 * jI[0])
            jQ[hilbertIdx] = (0.0962 * Q1[hilbertIdx]) + (0.5769 * jQ[1]) - (0.5769 * jQ[2]) - (0.0962 * jQ[0])
            hilbertIdx = (hilbertIdx + 1) % 3

            Q2 = (0.2 * (Q1[hilbertIdx] + jI[hilbertIdx])) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForEvenPrev3 - jQ[hilbertIdx])) + (0.8 * prevI2)

            I1ForOddPrev3 = I1ForOddPrev2
            I1ForOddPrev2 = detrender[hilbertIdx]

            if I1ForEvenPrev3 != 0.0:
                tempReal2 = math.atan(Q1[hilbertIdx] / I1ForEvenPrev3) * rad2Deg
            else:
                tempReal2 = 0.0
        else:
            detrender[hilbertIdx] = (0.0962 * smoothedValue) + (0.5769 * detrender[1]) - (0.5769 * detrender[2]) - (0.0962 * detrender[0])
            Q1[hilbertIdx] = (0.0962 * detrender[hilbertIdx]) + (0.5769 * Q1[1]) - (0.5769 * Q1[2]) - (0.0962 * Q1[0])
            jI[hilbertIdx] = (0.0962 * I1ForOddPrev3) + (0.5769 * jI[1]) - (0.5769 * jI[2]) - (0.0962 * jI[0])
            jQ[hilbertIdx] = (0.0962 * Q1[hilbertIdx]) + (0.5769 * jQ[1]) - (0.5769 * jQ[2]) - (0.0962 * jQ[0])

            Q2 = (0.2 * (Q1[hilbertIdx] + jI[hilbertIdx])) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForOddPrev3 - jQ[hilbertIdx])) + (0.8 * prevI2)

            I1ForEvenPrev3 = I1ForEvenPrev2
            I1ForEvenPrev2 = detrender[hilbertIdx]

            if I1ForOddPrev3 != 0.0:
                tempReal2 = math.atan(Q1[hilbertIdx] / I1ForOddPrev3) * rad2Deg
            else:
                tempReal2 = 0.0

        tempReal = prevPhase - tempReal2
        prevPhase = tempReal2
        if tempReal < 1.0:
            tempReal = 1.0

        if tempReal > 1.0:
            tempReal = optInFastLimit / tempReal
            if tempReal < optInSlowLimit:
                tempReal = optInSlowLimit
        else:
            tempReal = optInFastLimit

        mama = (tempReal * todayValue) + ((1 - tempReal) * mama)
        tempReal *= 0.5
        fama = (tempReal * mama) + ((1 - tempReal) * fama)

        if today >= startIdx:
            outMAMA[outIdx] = mama
            outFAMA[outIdx] = fama
            outIdx += 1

        Re = (0.2 * ((I2 * prevI2) + (Q2 * prevQ2))) + (0.8 * Re)
        Im = (0.2 * ((I2 * prevQ2) - (Q2 * prevI2))) + (0.8 * Im)
        prevQ2 = Q2
        prevI2 = I2
        tempReal = period
        if (Im != 0.0) and (Re != 0.0):
            period = 360.0 / (math.atan(Im / Re) * rad2Deg)
        tempReal2 = 1.5 * tempReal
        if period > tempReal2:
            period = tempReal2
        tempReal2 = 0.67 * tempReal
        if period < tempReal2:
            period = tempReal2
        if period < 6:
            period = 6
        elif period > 50:
            period = 50
        period = (0.2 * period) + (0.8 * tempReal)

        today += 1

    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def MAMA(real: np.ndarray, fastlimit: float = 0.5, slowlimit: float = 0.05):
    """MAMA(real, fastlimit=0.5, slowlimit=0.05)

    MESA Adaptive Moving Average

    Inputs:
        real: (any ndarray)
        fastlimit: (float) Upper limit use in the adaptive algorithm
        slowlimit: (float) Lower limit use in the adaptive algorithm
    Outputs:
        mama: (ndarray) MESA Adaptive Moving Average
        fama: (ndarray) Following Adaptive Moving Average
    """
    real = check_array(real)

    outMAMA = np.full_like(real, np.nan)
    outFAMA = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_MAMA_Lookback(fastlimit, slowlimit)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_MAMA(0, endIdx, real[startIdx:], fastlimit, slowlimit,
            outBegIdx, outNBElement, outMAMA[lookback:], outFAMA[lookback:])
    return outMAMA, outFAMA

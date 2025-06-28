import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from .hilbert_transform import HilbertVariable

if not cython.compiled:
    from math import atan
    from .hilbert_transform import do_odd, do_even
    from .ta_utility import TA_INTEGER_DEFAULT

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MAMA_Lookback(optInFastLimit: cython.double, optInSlowLimit: cython.double) -> cython.Py_ssize_t:
    """TA_MAMA_Lookback(optInFastLimit, optInSlowLimit) -> Py_ssize_t

    MAMA Lookback calculation function.
    
    The two parameters are not a factor to determine the lookback, 
    but are still requested for consistency with all other Lookback functions.
    
    Lookback is a fix amount + the unstable period:
    - 12 price bar for TradeStation compatibility
    - 6 price bars for the Detrender
    - 6 price bars for Q1
    - 3 price bars for jI
    - 3 price bars for jQ
    - 1 price bar for Re/Im
    - 1 price bar for the Delta Phase
    -------------------
    - 32 Total fixed lookback
    """
    # Range check for parameters
    # Default values and validation as per C implementation
    if optInFastLimit == TA_INTEGER_DEFAULT:
        optInFastLimit = 5.000000e-1
    elif (optInFastLimit < 1.000000e-2) or (optInFastLimit > 9.900000e-1):
        return -1

    if optInSlowLimit == TA_INTEGER_DEFAULT:
        optInSlowLimit = 5.000000e-2
    elif (optInSlowLimit < 1.000000e-2) or (optInSlowLimit > 9.900000e-1):
        return -1

    # Fixed lookback of 32 plus unstable period
    return 32 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MAMA)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
    """TA_MAMA - MESA Adaptive Moving Average calculation function.
    
    Input  = double
    Output = double, double (MAMA and FAMA)
    
    Optional Parameters:
    -------------------
    optInFastLimit:(From 0.01 to 0.99)
        Upper limit use in the adaptive algorithm
    optInSlowLimit:(From 0.01 to 0.99)
        Lower limit use in the adaptive algorithm
    """
    today: cython.Py_ssize_t = 0
    trailingWMAIdx: cython.Py_ssize_t = 0
    tempReal: cython.double = 0.0
    tempReal2: cython.double = 0.0
    periodWMASub: cython.double = 0.0
    periodWMASum: cython.double = 0.0
    trailingWMAValue: cython.double = 0.0
    smoothedValue: cython.double = 0.0
    
    # Parameter validation section
    if not TA_FUNC_NO_RANGE_CHECK:
        # Validate the requested output range
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        # Check input parameters
        if optInFastLimit == TA_INTEGER_DEFAULT:
            optInFastLimit = 5.000000e-1
        elif (optInFastLimit < 1.000000e-2) or (optInFastLimit > 9.900000e-1):
            return TA_RetCode.TA_BAD_PARAM

        if optInSlowLimit == TA_INTEGER_DEFAULT:
            optInSlowLimit = 5.000000e-2
        elif (optInSlowLimit < 1.000000e-2) or (optInSlowLimit > 9.900000e-1):
            return TA_RetCode.TA_BAD_PARAM

    # Constants initialization
    rad2Deg: cython.double = 180.0 / (4.0 * atan(1))
    lookbackTotal: cython.Py_ssize_t = 32 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MAMA)

    # Adjust start index if not enough initial data
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Check if there's data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx

    # Initialize price smoother (weighted moving average)
    trailingWMAIdx = startIdx - lookbackTotal
    today = trailingWMAIdx

    # Unrolled initialization for speed optimization
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

    # Macro-like implementation for WMA calculation
    # def DO_PRICE_WMA(new_price, smoothed_value):
    #     nonlocal periodWMASub, periodWMASum, trailingWMAValue, trailingWMAIdx
    #     periodWMASub += new_price
    #     periodWMASub -= trailingWMAValue
    #     periodWMASum += new_price * 4.0
    #     trailingWMAValue = inReal[trailingWMAIdx]
    #     trailingWMAIdx += 1
    #     smoothed_value = periodWMASum * 0.1
    #     periodWMASum -= periodWMASub
    #     return smoothed_value

    # Initialize WMA with unrolled loop
    i: cython.int = 9
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

    # Hilbert transform circular buffers initialization
    hilbertIdx: cython.int = 0

    # Hilbert transform variables (circular buffers of size 3)
    detrender: HilbertVariable = HilbertVariable()
    Q1: HilbertVariable = HilbertVariable()
    jI: HilbertVariable = HilbertVariable()
    jQ: HilbertVariable = HilbertVariable()

    # Constants for Hilbert transform
    a: cython.double = 0.0962
    b: cython.double = 0.5769

    # Main variables initialization
    period: cython.double = 0.0
    outIdx: cython.Py_ssize_t = 0


    Q2: cython.double = 0.0
    I2: cython.double = 0.0
    prevI2: cython.double = 0.0
    prevQ2: cython.double = 0.0
    Re: cython.double = 0.0
    Im: cython.double = 0.0
    mama: cython.double = 0.0
    fama: cython.double = 0.0
    I1ForOddPrev3: cython.double = 0.0
    I1ForEvenPrev3: cython.double = 0.0
    I1ForOddPrev2: cython.double = 0.0
    I1ForEvenPrev2: cython.double = 0.0

    prevPhase: cython.double = 0.0

    # Main calculation loop
    while today <= endIdx:
        adjustedPrevPeriod: cython.double = (0.075 * period) + 0.54

        todayValue: cython.double = inReal[today]
        
        periodWMASub += todayValue
        periodWMASub -= trailingWMAValue
        periodWMASum += todayValue * 4.0
        trailingWMAValue = inReal[trailingWMAIdx]
        trailingWMAIdx += 1
        smoothedValue = periodWMASum * 0.1
        periodWMASum -= periodWMASub

        if (today % 2) == 0:
            # Hilbert Transforms for even price bar
            do_even(detrender, smoothedValue, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(Q1, detrender.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(jI, I1ForEvenPrev3, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(jQ, Q1.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
 
            hilbertIdx += 1
            if hilbertIdx == 3:
                hilbertIdx = 0

            Q2 = (0.2 * (Q1.current_value + jI.current_value)) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForEvenPrev3 - jQ.current_value)) + (0.8 * prevI2)

            # Save detrender for odd logic
            I1ForOddPrev3 = I1ForOddPrev2
            I1ForOddPrev2 = detrender.current_value

            # Calculate Alpha
            if I1ForEvenPrev3 != 0.0:
                tempReal2 = atan(Q1.current_value / I1ForEvenPrev3) * rad2Deg
            else:
                tempReal2 = 0.0
        else:
            # Hilbert Transforms for odd price bar
            do_odd(detrender, smoothedValue, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(Q1, detrender.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(jI, I1ForOddPrev3, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(jQ, Q1.current_value, hilbertIdx, a, b, adjustedPrevPeriod)

            Q2 = (0.2 * (Q1.current_value + jI.current_value)) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForOddPrev3 - jQ.current_value)) + (0.8 * prevI2)

            # Save detrender for even logic
            I1ForEvenPrev3 = I1ForEvenPrev2
            I1ForEvenPrev2 = detrender.current_value

            # Calculate Alpha
            if I1ForOddPrev3 != 0.0:
                tempReal2 = atan(Q1.current_value / I1ForOddPrev3) * rad2Deg
            else:
                tempReal2 = 0.0

        # Calculate Delta Phase
        tempReal = prevPhase - tempReal2
        prevPhase = tempReal2
        if tempReal < 1.0:
            tempReal = 1.0

        # Calculate adaptive factor
        if tempReal > 1.0:
            tempReal = optInFastLimit / tempReal
            if tempReal < optInSlowLimit:
                tempReal = optInSlowLimit
        else:
            tempReal = optInFastLimit

        # Calculate MAMA and FAMA
        mama = (tempReal * todayValue) + ((1 - tempReal) * mama)
        tempReal *= 0.5
        fama = (tempReal * mama) + ((1 - tempReal) * fama)

        # Store results if within valid range
        if today >= startIdx:
            outMAMA[outIdx] = mama
            outFAMA[outIdx] = fama
            outIdx += 1

        # Adjust period for next calculation
        Re = (0.2 * ((I2 * prevI2) + (Q2 * prevQ2))) + (0.8 * Re)
        Im = (0.2 * ((I2 * prevQ2) - (Q2 * prevI2))) + (0.8 * Im)
        prevQ2 = Q2
        prevI2 = I2
        
        tempReal = period
        if (Im != 0.0) and (Re != 0.0):
            period = 360.0 / (atan(Im / Re) * rad2Deg)
        
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

def MAMA(
    real: np.ndarray, 
    fastlimit: float = 0.5, 
    slowlimit: float = 0.05,
):
    """MAMA(real, fastlimit=0.5, slowlimit=0.05)

    MESA Adaptive Moving Average calculation.
    
    This function calculates the MESA Adaptive Moving Average (MAMA) and 
    the Following Adaptive Moving Average (FAMA).
    
    Inputs:
        real: (any ndarray) Input price data
        fastlimit: (float) Upper limit for adaptive algorithm (0.01 to 0.99)
        slowlimit: (float) Lower limit for adaptive algorithm (0.01 to 0.99)
    
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

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_MAMA(
        0, endIdx, real[startIdx:], fastlimit, slowlimit,
        outBegIdx, outNBElement, outMAMA[lookback:], outFAMA[lookback:],
    )
    return outMAMA, outFAMA
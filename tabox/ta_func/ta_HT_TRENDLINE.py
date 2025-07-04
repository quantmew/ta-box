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
def TA_HT_TRENDLINE_Lookback() -> cython.Py_ssize_t:
    """
    TA_HT_TRENDLINE_Lookback - Hilbert Transform - Instantaneous Trendline Lookback

    Returns:
        Number of lookback periods
    """
    # 31 inputs are skipped 
    # +32 outputs are skipped to account for misc lookback
    # ---
    # 63 Total Lookback
    #
    # 31 is for compatibility with Tradestation.
    # See TA_MAMA_Lookback for an explanation of the "32".
    return 63 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_HT_TRENDLINE)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_HT_TRENDLINE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline

    Input  = double
    Output = double
    """
    # Local variables
    today: cython.Py_ssize_t
    trailingWMAIdx: cython.Py_ssize_t
    tempReal: cython.double
    tempReal2: cython.double
    periodWMASub: cython.double
    periodWMASum: cython.double
    trailingWMAValue: cython.double
    smoothedValue: cython.double
    iTrend1: cython.double
    iTrend2: cython.double
    iTrend3: cython.double
    a: cython.double
    b: cython.double
    hilbertTempReal: cython.double
    hilbertIdx: cython.int
    Q2: cython.double
    I2: cython.double
    prevQ2: cython.double
    prevI2: cython.double
    Re: cython.double
    Im: cython.double
    I1ForOddPrev2: cython.double
    I1ForOddPrev3: cython.double
    I1ForEvenPrev2: cython.double
    I1ForEvenPrev3: cython.double
    rad2Deg: cython.double
    todayValue: cython.double
    smoothPeriod: cython.double
    SMOOTH_PRICE_SIZE: cython.int = 50
    smoothPrice: cython.double[50]  # Circular buffer
    smoothPrice_Idx: cython.int
    idx: cython.Py_ssize_t
    DCPeriod: cython.double
    DCPeriodInt: cython.int
    i: cython.int
    outIdx: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t

    # Parameter validation
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Initialize circular buffer for smoothPrice
    for i in range(SMOOTH_PRICE_SIZE):
        smoothPrice[i] = 0.0
    smoothPrice_Idx = 0

    # Initialize trendline variables
    iTrend1 = iTrend2 = iTrend3 = 0.0

    # Calculate radian to degree conversion factor
    tempReal = atan(1.0)
    rad2Deg = 45.0 / tempReal

    # Calculate lookback period
    lookbackTotal = 63 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_HT_TRENDLINE)

    # Adjust start index if insufficient data
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Check if output is possible
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx

    # Initialize price smoother (weighted moving average)
    trailingWMAIdx = startIdx - lookbackTotal
    today = trailingWMAIdx

    # Initialize WMA components with unrolled loop
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

    # Complete WMA initialization with remaining 34 iterations
    i = 34
    while i != 0:
        tempReal = inReal[today]
        today += 1
        # Update WMA components
        periodWMASub += tempReal
        periodWMASub -= trailingWMAValue
        periodWMASum += tempReal * 4.0
        trailingWMAValue = inReal[trailingWMAIdx]
        trailingWMAIdx += 1
        smoothedValue = periodWMASum * 0.1
        periodWMASum -= periodWMASub
        i -= 1

    # Initialize Hilbert transform variables
    hilbertIdx = 0
    detrender = HilbertVariable()
    Q1 = HilbertVariable()
    jI = HilbertVariable()
    jQ = HilbertVariable()
    a = 0.0962
    b = 0.5769

    # Main calculation variables
    period = 0.0
    outIdx = 0
    prevI2 = prevQ2 = 0.0
    Re = Im = 0.0
    I1ForOddPrev3 = I1ForEvenPrev3 = 0.0
    I1ForOddPrev2 = I1ForEvenPrev2 = 0.0
    smoothPeriod = 0.0

    # Main calculation loop
    while today <= endIdx:
        adjustedPrevPeriod = (0.075 * period) + 0.54

        todayValue = inReal[today]
        # Update price smoother (WMA)
        periodWMASub += todayValue
        periodWMASub -= trailingWMAValue
        periodWMASum += todayValue * 4.0
        trailingWMAValue = inReal[trailingWMAIdx]
        trailingWMAIdx += 1
        smoothedValue = periodWMASum * 0.1
        periodWMASum -= periodWMASub

        # Store smoothed value in circular buffer
        smoothPrice[smoothPrice_Idx] = smoothedValue

        if (today % 2) == 0:
            # Hilbert Transform for even price bar
            do_even(detrender, smoothedValue, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(Q1, detrender.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(jI, I1ForEvenPrev3, hilbertIdx, a, b, adjustedPrevPeriod)
            do_even(jQ, Q1.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
            
            hilbertIdx += 1
            if hilbertIdx == 3:
                hilbertIdx = 0

            Q2 = (0.2 * (Q1.current_value + jI.current_value)) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForEvenPrev3 - jQ.current_value)) + (0.8 * prevI2)

            # Update delayed values for odd cycles
            I1ForOddPrev3 = I1ForOddPrev2
            I1ForOddPrev2 = detrender.current_value
        else:
            # Hilbert Transform for odd price bar
            do_odd(detrender, smoothedValue, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(Q1, detrender.current_value, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(jI, I1ForOddPrev3, hilbertIdx, a, b, adjustedPrevPeriod)
            do_odd(jQ, Q1.current_value, hilbertIdx, a, b, adjustedPrevPeriod)

            Q2 = (0.2 * (Q1.current_value + jI.current_value)) + (0.8 * prevQ2)
            I2 = (0.2 * (I1ForOddPrev3 - jQ.current_value)) + (0.8 * prevI2)

            # Update delayed values for even cycles
            I1ForEvenPrev3 = I1ForEvenPrev2
            I1ForEvenPrev2 = detrender.current_value

        # Update period calculation
        Re = (0.2 * ((I2 * prevI2) + (Q2 * prevQ2))) + (0.8 * Re)
        Im = (0.2 * ((I2 * prevQ2) - (Q2 * prevI2))) + (0.8 * Im)
        prevQ2 = Q2
        prevI2 = I2

        tempReal = period
        if Im != 0.0 and Re != 0.0:
            period = 360.0 / (atan(Im / Re) * rad2Deg)
        
        # Constrain period values
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

        smoothPeriod = (0.33 * period) + (0.67 * smoothPeriod)

        # Calculate Instantaneous Trendline
        DCPeriod = smoothPeriod + 0.5
        DCPeriodInt = cython.cast(cython.int, DCPeriod)

        # Sum recent prices
        idx = today
        tempReal = 0.0
        for i in range(DCPeriodInt):
            tempReal += inReal[idx]
            idx -= 1
        
        # Average calculation
        if DCPeriodInt > 0:
            tempReal /= DCPeriodInt
        
        # Update trendline components
        tempReal2 = (4.0 * tempReal + 3.0 * iTrend1 + 2.0 * iTrend2 + iTrend3) / 10.0
        iTrend3 = iTrend2
        iTrend2 = iTrend1
        iTrend1 = tempReal

        # Store output if in valid range
        if today >= startIdx:
            outReal[outIdx] = tempReal2
            outIdx += 1

        # Prepare for next iteration
        smoothPrice_Idx = (smoothPrice_Idx + 1) % SMOOTH_PRICE_SIZE
        today += 1

    # Set output metadata
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def HT_TRENDLINE(real: np.ndarray) -> np.ndarray:
    """
    HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline (Overlap Studies)

    Inputs:
        real: (any ndarray) Input price data
    Outputs:
        real: Hilbert Transform - Instantaneous Trendline values
    """
    real = check_array(real)
    length: cython.Py_ssize_t = real.shape[0]
    
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_HT_TRENDLINE_Lookback()

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_HT_TRENDLINE(
        0, endIdx, real[startIdx:], outBegIdx, outNBElement, outReal[lookback:]
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    
    return outReal
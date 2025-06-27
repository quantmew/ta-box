import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_IS_ZERO(v: cython.double) -> cython.bint:
    return ((-0.00000001) < v) and (v < 0.00000001)

def TA_BETA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod < 1 or optInTimePeriod > 100000:
            return -1
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 5

    return optInTimePeriod

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_BETA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal0: cython.double[::1],
    inReal1: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_BETA - Beta
    
    Input  = double, double
    Output = double
    
    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # Parameter check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal0 is None or inReal1 is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM
        
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 5
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

    # Algorithm variable definition
    S_xx: cython.double = 0.0  # sum of x * x
    S_xy: cython.double = 0.0  # sum of x * y
    S_x: cython.double = 0.0   # sum of x
    S_y: cython.double = 0.0   # sum of y
    last_price_x: cython.double = 0.0  # last price from inReal0
    last_price_y: cython.double = 0.0  # last price from inReal1
    trailing_last_price_x: cython.double = 0.0  # used to remove elements from trailing sum
    trailing_last_price_y: cython.double = 0.0  # used to remove elements from trailing sum
    tmp_real: cython.double = 0.0  # temporary variable
    x: cython.double  # 'x' value, last change between prices in inReal0
    y: cython.double  # 'y' value, last change between prices in inReal1
    n: cython.double = 0.0
    i: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t = 0
    trailingIdx: cython.Py_ssize_t
    nbInitialElementNeeded: cython.Py_ssize_t

    """
    Algorithm description:
    The Beta algorithm is a measure of a stock's volatility relative to an index. The stock prices
    are given in inReal0 and the index prices are given in inReal1. The sizes of these vectors
    should be equal. The algorithm is to calculate the change between prices in both vectors
    and then 'plot' these changes
    Beta of 1 means the stock changes exactly in sync with the market. A Beta less than 1 means
    the stock changes less than the market, and a Beta greater than 1 means the stock changes
    more than the market. The related value is the Alpha value (see TA_ALPHA), which is the
    Y-intercept of the same linear regression.
    """

    # Validate calculation method type and determine the minimum number of input elements needed before outputting the first value
    nbInitialElementNeeded = optInTimePeriod

    # If there is not enough initial data, move the start index up
    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    # Ensure there is still data to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Consume the first input
    trailingIdx = startIdx - nbInitialElementNeeded
    last_price_x = trailing_last_price_x = inReal0[trailingIdx]
    last_price_y = trailing_last_price_y = inReal1[trailingIdx]

    # Process the remaining lookback period until the first value is ready to be output
    i = trailingIdx + 1
    trailingIdx += 1
    while i < startIdx:
        tmp_real = inReal0[i]
        if not TA_IS_ZERO(last_price_x):
            x = (tmp_real - last_price_x) / last_price_x
        else:
            x = 0.0
        last_price_x = tmp_real

        tmp_real = inReal1[i]
        i += 1
        if not TA_IS_ZERO(last_price_y):
            y = (tmp_real - last_price_y) / last_price_y
        else:
            y = 0.0
        last_price_y = tmp_real

        S_xx += x * x
        S_xy += x * y
        S_x += x
        S_y += y

    # Start calculating and filling the output array
    outIdx = 0  # The first output always starts from index 0
    n = optInTimePeriod

    while True:
        tmp_real = inReal0[i]
        if not TA_IS_ZERO(last_price_x):
            x = (tmp_real - last_price_x) / last_price_x
        else:
            x = 0.0
        last_price_x = tmp_real

        tmp_real = inReal1[i]
        i += 1
        if not TA_IS_ZERO(last_price_y):
            y = (tmp_real - last_price_y) / last_price_y
        else:
            y = 0.0
        last_price_y = tmp_real

        S_xx += x * x
        S_xy += x * y
        S_x += x
        S_y += y

        # Always read trailing data before writing output, because the input and output buffers can be the same
        tmp_real = inReal0[trailingIdx]
        if not TA_IS_ZERO(trailing_last_price_x):
            x = (tmp_real - trailing_last_price_x) / trailing_last_price_x
        else:
            x = 0.0
        trailing_last_price_x = tmp_real

        tmp_real = inReal1[trailingIdx]
        trailingIdx += 1
        if not TA_IS_ZERO(trailing_last_price_y):
            y = (tmp_real - trailing_last_price_y) / trailing_last_price_y
        else:
            y = 0.0
        trailing_last_price_y = tmp_real

        # Write output
        tmp_real = (n * S_xx) - (S_x * S_x)
        if not TA_IS_ZERO(tmp_real):
            outReal[outIdx] = ((n * S_xy) - (S_x * S_y)) / tmp_real
        else:
            outReal[outIdx] = 0.0
        outIdx += 1

        # Remove calculations from trailing index
        S_xx -= x * x
        S_xy -= x * y
        S_x -= x
        S_y -= y

        if i > endIdx:
            break

    # All done. Indicate output limits and return
    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def BETA(real0: np.ndarray, real1: np.ndarray, timeperiod: int = 5) -> np.ndarray:
    """BETA(real0, real1[, timeperiod=5])
    
    Beta (Momentum Indicators)
    
    The Beta 'algorithm' is a measure of a stocks volatility vs from index. The stock prices
    are given in real0 and the index prices are give in real1. The size of these vectors
    should be equal. The algorithm is to calculate the change between prices in both vectors
    and then 'plot' these changes are points in the Euclidean plane. The x value of the point
    is market return and the y value is the security return. The beta value is the slope of a
    linear regression through these points. A beta of 1 is simple the line y=x, so the stock
    varies percisely with the market. A beta of less than one means the stock varies less than
    the market and a beta of more than one means the stock varies more than market. A related
    value is the Alpha value (see TA_ALPHA) which is the Y-intercept of the same linear regression.
    
    Inputs:
        real0: (any ndarray) Input series for stock prices
        real1: (any ndarray) Input series for index prices
    Parameters:
        timeperiod: 5 Number of periods
    Outputs:
        real
    """
    real0 = check_array(real0)
    real1 = check_array(real1)
    
    # Ensure the lengths of the two input arrays are the same
    if real0.shape[0] != real1.shape[0]:
        raise ValueError("Input arrays real0 and real1 must have the same length")
    
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real0.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real0)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + timeperiod  # TA_BETA_Lookback returns timeperiod directly

    outReal = np.full_like(real0, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_BETA(
        0,
        endIdx,
        real0[startIdx:],
        real1[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
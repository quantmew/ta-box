import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_LINEARREG_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_LINEARREG_Lookback - Linear Regression Lookback

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
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_LINEARREG(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_LINEARREG - Linear Regression

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    SumX: cython.double
    SumXY: cython.double
    SumY: cython.double
    SumXSqr: cython.double
    Divisor: cython.double
    m: cython.double
    b: cython.double
    i: cython.Py_ssize_t
    tempValue1: cython.double

    """
    Linear Regression is a concept also known as the
    "least squares method" or "best fit." Linear
    Regression attempts to fit a straight line between
    several data points in such a way that distance
    between each data point and the line is minimized.

    For each point, a straight line over the specified
    previous bar period is determined in terms
    of y = b + m*x:

    TA_LINEARREG          : Returns b+m*(period-1)
    TA_LINEARREG_SLOPE    : Returns 'm'
    TA_LINEARREG_ANGLE    : Returns 'm' in degree.
    TA_LINEARREG_INTERCEPT: Returns 'b'
    TA_TSF                : Returns b+m*(period)
    """

    # 调整startIdx以考虑回溯期
    lookbackTotal = TA_LINEARREG_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 确保仍有可评估的内容
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    today = startIdx

    # 预计算固定值
    SumX = optInTimePeriod * (optInTimePeriod - 1) * 0.5
    SumXSqr = optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6
    Divisor = SumX * SumX - optInTimePeriod * SumXSqr

    # 主要计算循环
    while today <= endIdx:
        SumXY = 0.0
        SumY = 0.0
        for i in range(optInTimePeriod - 1, -1, -1):
            tempValue1 = inReal[today - i]
            SumY += tempValue1
            SumXY += cython.cast(cython.double, i) * tempValue1

        # 计算斜率和截距
        m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor
        b = (SumY - m * SumX) / cython.cast(cython.double, optInTimePeriod)

        # 计算线性回归值并存储
        outReal[outIdx] = b + m * cython.cast(cython.double, optInTimePeriod - 1)
        outIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


def LINEARREG(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    LINEARREG(real[, timeperiod=14])

    Linear Regression (Overlap Studies)

    Linear Regression attempts to fit a straight line between several data points
    in such a way that distance between each data point and the line is minimized.
    The function returns b + m * (timeperiod - 1), where y = b + m*x is the regression line.

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
    lookback: cython.Py_ssize_t = startIdx + TA_LINEARREG_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_LINEARREG(
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

import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
import math

PI = math.pi


def TA_LINEARREG_ANGLE_Lookback(
    optInTimePeriod: cython.int,
) -> cython.Py_ssize_t:
    """
    TA_LINEARREG_ANGLE_Lookback - Linear Regression Angle Lookback

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
def TA_LINEARREG_ANGLE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_LINEARREG_ANGLE - Linear Regression Angle

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

    # 计算回溯期
    lookbackTotal: cython.Py_ssize_t = TA_LINEARREG_ANGLE_Lookback(optInTimePeriod)

    # 调整起始索引以考虑回溯期
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 确保有数据可计算
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx: cython.Py_ssize_t = 0  # 输出索引
    today: cython.Py_ssize_t = startIdx

    # 预计算常数项
    SumX: cython.double = optInTimePeriod * (optInTimePeriod - 1) * 0.5
    SumXSqr: cython.double = (
        optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6
    )
    Divisor: cython.double = SumX * SumX - optInTimePeriod * SumXSqr

    # 主计算循环
    while today <= endIdx:
        SumXY: cython.double = 0.0
        SumY: cython.double = 0.0
        for i in range(optInTimePeriod - 1, -1, -1):
            tempValue1: cython.double = inReal[today - i]
            SumY += tempValue1
            SumXY += cython.cast(cython.double, i) * tempValue1

        # 计算斜率 m
        m: cython.double = (optInTimePeriod * SumXY - SumX * SumY) / Divisor
        # 计算角度并转换为度数
        outReal[outIdx] = math.atan(m) * (180.0 / PI)
        outIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def LINEARREG_ANGLE(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    LINEARREG_ANGLE(real[, timeperiod=14])

    Linear Regression Angle (Overlap Studies)

    The linear regression angle is calculated using the least squares method to
    fit a straight line to the data points over a specified period, and then
    converting the slope of that line to an angle in degrees.

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
    lookback: cython.Py_ssize_t = startIdx + TA_LINEARREG_ANGLE_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_LINEARREG_ANGLE(
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

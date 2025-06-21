import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_OBV_Lookback() -> cython.Py_ssize_t:
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_OBV(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    inVolume: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_OBV - On Balance Volume
    
    Input  = double (价格), double (成交量)
    Output = double (OBV值)
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal is None or inVolume is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # 初始化变量
    prevOBV: cython.double = inVolume[startIdx]
    prevReal: cython.double = inReal[startIdx]
    outIdx: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t

    # 计算OBV
    for i in range(startIdx, endIdx + 1):
        tempReal: cython.double = inReal[i]
        if tempReal > prevReal:
            prevOBV += inVolume[i]
        elif tempReal < prevReal:
            prevOBV -= inVolume[i]
        # 价格不变时OBV不变
        
        outReal[outIdx] = prevOBV
        outIdx += 1
        prevReal = tempReal

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def OBV(real: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV(real, volume)
    
    On Balance Volume (量价指标)
    
    OBV通过累计成交量来确认价格趋势：
    - 价格上涨时，成交量加到OBV
    - 价格下跌时，成交量从OBV中减去
    - 价格不变时，OBV保持不变
    
    Inputs:
        real: (np.ndarray) 价格序列
        volume: (np.ndarray) 成交量序列，必须与价格序列长度相同
    Outputs:
        np.ndarray: OBV指标值序列
    """
    # 检查输入数组
    real = check_array(real)
    volume = check_array(volume)
    
    if real.shape[0] != volume.shape[0]:
        raise ValueError("价格序列和成交量序列长度必须相同")

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_OBV(
        0,
        endIdx,
        real[startIdx:],
        volume[startIdx:],
        outBegIdx,
        outNBElement,
        outReal[startIdx:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
# define TA_REAL_EQ(x,v,ep)   (((v-ep)<x)&&(x<(v+ep)))
# define TA_IS_ZERO(v)        (((-0.00000001)<v)&&(v<0.00000001))
# define TA_IS_ZERO_OR_NEG(v) (v<0.00000001)
import cython
import math
from typing import List
from ..retcode import TA_RetCode


# 数学常量
PI = 3.14159265358979323846


def TA_REAL_EQ(x: cython.float, v: cython.float, ep: cython.float):
    return ((v - ep) < x) and (x < (v + ep))


def TA_IS_ZERO(v: cython.float):
    return ((-0.00000001) < v) and (v < 0.00000001)


def TA_IS_ZERO_OR_NEG(v: cython.float):
    return v < 0.00000001


# 数学函数
def std_floor(x: cython.float) -> cython.float:
    return math.floor(x)


def std_ceil(x: cython.float) -> cython.float:
    return math.ceil(x)


def std_fabs(x: cython.float) -> cython.float:
    return abs(x)


def std_atan(x: cython.float) -> cython.float:
    return math.atan(x)


def std_cos(x: cython.float) -> cython.float:
    return math.cos(x)


def std_sin(x: cython.float) -> cython.float:
    return math.sin(x)


def std_sqrt(x: cython.float) -> cython.float:
    return math.sqrt(x)


def std_tanh(x: cython.float) -> cython.float:
    return math.tanh(x)


def std_tan(x: cython.float) -> cython.float:
    return math.tan(x)


def std_sinh(x: cython.float) -> cython.float:
    return math.sinh(x)


def std_log10(x: cython.float) -> cython.float:
    return math.log10(x)


def std_log(x: cython.float) -> cython.float:
    return math.log(x)


def std_exp(x: cython.float) -> cython.float:
    return math.exp(x)


def std_cosh(x: cython.float) -> cython.float:
    return math.cosh(x)


def std_asin(x: cython.float) -> cython.float:
    return math.asin(x)


def std_acos(x: cython.float) -> cython.float:
    return math.acos(x)


# Utility Functions
def round_pos(x: cython.float) -> cython.float:
    return std_floor(x + 0.5)


def round_neg(x: cython.float) -> cython.float:
    return std_ceil(x - 0.5)


def round_pos_2(x: cython.float) -> cython.float:
    return (std_floor((x * 100.0) + 0.5)) / 100.0


def round_neg_2(x: cython.float) -> cython.float:
    return (std_ceil((x * 100.0) - 0.5)) / 100.0


def PER_TO_K(per: cython.int) -> cython.float:
    return 2.0 / (per + 1)


# K-Line Related Functions
def TA_REALBODY(inClose: cython.float, inOpen: cython.float) -> cython.float:
    return std_fabs(inClose - inOpen)


def TA_UPPERSHADOW(
    inHigh: cython.float, inClose: cython.float, inOpen: cython.float
) -> cython.float:
    return inHigh - (inClose if inClose >= inOpen else inOpen)


def TA_LOWERSHADOW(
    inClose: cython.float, inOpen: cython.float, inLow: cython.float
) -> cython.float:
    return (inOpen if inClose >= inOpen else inClose) - inLow


def TA_HIGHLOWRANGE(inHigh: cython.float, inLow: cython.float) -> cython.float:
    return inHigh - inLow


def TA_CANDLECOLOR(inClose: cython.float, inOpen: cython.float) -> cython.int:
    return 1 if inClose >= inOpen else -1


from enum import Enum, IntEnum


class TA_FuncUnstId(IntEnum):
    TA_FUNC_UNST_SMA = 0
    TA_FUNC_UNST_EMA = 1
    TA_FUNC_UNST_WMA = 2
    TA_FUNC_UNST_DEMA = 3
    TA_FUNC_UNST_TEMA = 4
    TA_FUNC_UNST_TRIMA = 5
    TA_FUNC_UNST_KAMA = 6
    TA_FUNC_UNST_MAMA = 7
    TA_FUNC_UNST_T3 = 8
    TA_FUNC_UNST_MA = 9
    TA_FUNC_UNST_MACD = 10
    TA_FUNC_UNST_MACD_SIGNAL = 11
    TA_FUNC_UNST_MACD_HIST = 12
    TA_FUNC_UNST_STOCH = 13
    TA_FUNC_UNST_STOCHF = 14
    TA_FUNC_UNST_ROC = 15
    TA_FUNC_UNST_ROCP = 16
    TA_FUNC_UNST_ROCR = 17
    TA_FUNC_UNST_ROCR100 = 18
    TA_FUNC_UNST_TRIX = 19
    TA_FUNC_UNST_ULTOSC = 20
    TA_FUNC_UNST_RSI = 21
    TA_FUNC_UNST_STOCHRSI = 22
    TA_FUNC_UNST_WILLR = 23
    TA_FUNC_UNST_ADX = 24
    TA_FUNC_UNST_ADXR = 25
    TA_FUNC_UNST_APO = 26
    TA_FUNC_UNST_PPO = 27
    TA_FUNC_UNST_MOM = 28
    TA_FUNC_UNST_BBANDS_UPPER = 29
    TA_FUNC_UNST_BBANDS_MIDDLE = 30
    TA_FUNC_UNST_BBANDS_LOWER = 31
    TA_FUNC_UNST_ATR = 32
    TA_FUNC_UNST_NATR = 33
    TA_FUNC_UNST_TRANGE = 34
    TA_FUNC_UNST_AROON_UP = 35
    TA_FUNC_UNST_AROON_DOWN = 36
    TA_FUNC_UNST_AROON_OSC = 37
    TA_FUNC_UNST_MFI = 38
    TA_FUNC_UNST_OBV = 39
    TA_FUNC_UNST_CCI = 40
    TA_FUNC_UNST_AD = 41
    TA_FUNC_UNST_ADOSC = 42
    TA_FUNC_UNST_ONVOLO = 43
    TA_FUNC_UNST_ALL = 44


class TA_Compatibility(IntEnum):
    TA_COMPATIBILITY_DEFAULT = 0
    TA_COMPATIBILITY_STANDARD = 1
    TA_COMPATIBILITY_METASTOCK = 2
    TA_COMPATIBILITY_TA_LIB = 3


class TA_Globals_t:
    unstablePeriod: List[cython.int]
    compatibility: TA_Compatibility

    def __init__(self):
        self.unstablePeriod = [0] * TA_FuncUnstId.TA_FUNC_UNST_ALL
        self.compatibility = TA_Compatibility.TA_COMPATIBILITY_DEFAULT


TA_Globals = TA_Globals_t()


def TA_GLOBALS_UNSTABLE_PERIOD(id: TA_FuncUnstId) -> cython.int:
    return TA_Globals.unstablePeriod[id]


def TA_GLOBALS_COMPATIBILITY() -> TA_Compatibility:
    return TA_Globals.compatibility


def TA_SetUnstablePeriod(id: TA_FuncUnstId, unstablePeriod: int) -> TA_RetCode:
    if id > TA_FuncUnstId.TA_FUNC_UNST_ALL:
        return TA_RetCode.TA_BAD_PARAM

    if id == TA_FuncUnstId.TA_FUNC_UNST_ALL:
        for i in range(TA_FuncUnstId.TA_FUNC_UNST_ALL):
            TA_Globals.unstablePeriod[i] = unstablePeriod
    else:
        TA_Globals.unstablePeriod[id] = unstablePeriod

    return TA_RetCode.TA_SUCCESS


def TA_GetUnstablePeriod(id: TA_FuncUnstId) -> int:
    if id >= TA_FuncUnstId.TA_FUNC_UNST_ALL:
        return 0

    return TA_Globals.unstablePeriod[id]


def TA_SetCompatibility(value: TA_Compatibility) -> TA_RetCode:
    TA_Globals.compatibility = value
    return TA_RetCode.TA_SUCCESS


def TA_GetCompatibility() -> TA_Compatibility:
    return TA_Globals.compatibility

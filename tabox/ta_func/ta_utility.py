#define TA_REAL_EQ(x,v,ep)   (((v-ep)<x)&&(x<(v+ep)))
#define TA_IS_ZERO(v)        (((-0.00000001)<v)&&(v<0.00000001))
#define TA_IS_ZERO_OR_NEG(v) (v<0.00000001)
import cython
import math

# 数学常量
PI = 3.14159265358979323846

def TA_REAL_EQ(x: cython.float, v: cython.float, ep: cython.float):
    return ((v-ep)<x) and (x<(v+ep))

def TA_IS_ZERO(v: cython.float):
    return ((-0.00000001)<v) and (v<0.00000001)

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

# 辅助函数
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

# K线图相关函数
def TA_REALBODY(inClose: cython.float, inOpen: cython.float) -> cython.float:
    return std_fabs(inClose - inOpen)

def TA_UPPERSHADOW(inHigh: cython.float, inClose: cython.float, inOpen: cython.float) -> cython.float:
    return inHigh - (inClose if inClose >= inOpen else inOpen)

def TA_LOWERSHADOW(inClose: cython.float, inOpen: cython.float, inLow: cython.float) -> cython.float:
    return (inOpen if inClose >= inOpen else inClose) - inLow

def TA_HIGHLOWRANGE(inHigh: cython.float, inLow: cython.float) -> cython.float:
    return inHigh - inLow

def TA_CANDLECOLOR(inClose: cython.float, inOpen: cython.float) -> cython.int:
    return 1 if inClose >= inOpen else -1
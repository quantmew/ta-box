#define TA_REAL_EQ(x,v,ep)   (((v-ep)<x)&&(x<(v+ep)))
#define TA_IS_ZERO(v)        (((-0.00000001)<v)&&(v<0.00000001))
#define TA_IS_ZERO_OR_NEG(v) (v<0.00000001)
import cython

def TA_REAL_EQ(x: cython.float, v: cython.float, ep: cython.float):
    return ((v-ep)<x) and (x<(v+ep))

def TA_IS_ZERO(v: cython.float):
    return ((-0.00000001)<v) and (v<0.00000001)

def TA_IS_ZERO_OR_NEG(v: cython.float):
    return v < 0.00000001
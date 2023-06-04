import numpy
import tabox
import talib


# close = numpy.random.random(100)


close = numpy.array([1,2,3,4,5,6,7,8,9], dtype=numpy.float64)
# output = tabox.SMA(close, timeperiod=3)


output = tabox.WMA(close, 3)
print(output)
output = talib.WMA(close, 3)
print(output)
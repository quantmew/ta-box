
# min/max value for a TA_Integer
TA_INTEGER_MIN = -2147483647  # INT_MIN+1 typically (-2^31 + 1)
TA_INTEGER_MAX = 2147483647   # INT_MAX typically (2^31 - 1)

# min/max value for a TA_Real
# Use fix value making sense in the context of TA-Lib (avoid to use float('inf')
# because they may cause issues in some contexts)
TA_REAL_MIN = -3e37
TA_REAL_MAX = 3e37

# A value outside of the min/max range indicates an undefined or default value.
TA_INTEGER_DEFAULT = -2147483648  # INT_MIN typically (-2^31)
TA_REAL_DEFAULT = -4e37

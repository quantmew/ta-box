
import time
import numpy as np
from typing import Callable

def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func


def bench(func:Callable, repeat:int=5):
    """
        decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        time_list = []
        for i in range(repeat):
            t1 = time.time()
            r = func(*args, **keyArgs)
            t2 = time.time()
            time_list.append(t2 - t1)

        print("Function=%s, TotalTime=%s, MaxTime=%s, MinTime=%s, MeanTime=%s" % (
            func.__name__,
            sum(time_list),
            np.max(time_list),
            np.min(time_list),
            np.mean(time_list),
            )
        )
        return r

    return st_func
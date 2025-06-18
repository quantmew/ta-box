# cython: language_level=3
import cython
from typing import List, Union


class HilbertVariable:
    def __init__(self):
        self.odd: Union[List[cython.double], cython.double[::1]] = [0.0, 0.0, 0.0]
        self.even: Union[List[cython.double], cython.double[::1]] = [0.0, 0.0, 0.0]
        self.current_value: cython.double = 0.0
        self.prev_odd: cython.double = 0.0
        self.prev_even: cython.double = 0.0
        self.prev_input_odd: cython.double = 0.0
        self.prev_input_even: cython.double = 0.0
    
    @property
    def value(self) -> cython.double:
        return self.current_value

    def do_transform(
        self,
        input_value: cython.double,
        hilbert_idx: cython.int,
        a: cython.double,
        b: cython.double,
        adjusted_prev_period: cython.double,
        is_odd: cython.bint,
    ):
        hilbert_temp_real: cython.double = a * input_value

        if is_odd:
            self.current_value = -self.odd[hilbert_idx]
            self.odd[hilbert_idx] = hilbert_temp_real
            self.current_value += hilbert_temp_real
            self.current_value -= self.prev_odd
            self.prev_odd = b * self.prev_input_odd
            self.current_value += self.prev_odd
            self.prev_input_odd = input_value
        else:
            self.current_value = -self.even[hilbert_idx]
            self.even[hilbert_idx] = hilbert_temp_real
            self.current_value += hilbert_temp_real
            self.current_value -= self.prev_even
            self.prev_even = b * self.prev_input_even
            self.current_value += self.prev_even
            self.prev_input_even = input_value

        self.current_value *= adjusted_prev_period

def do_odd(
    hilbert_variable: HilbertVariable,
    input_value: cython.double,
    hilbert_idx: cython.int,
    a: cython.double,
    b: cython.double,
    adjusted_prev_period: cython.double,
):
    hilbert_variable.do_transform(input_value, hilbert_idx, a, b, adjusted_prev_period, True)

def do_even(
    hilbert_variable: HilbertVariable,
    input_value: cython.double,
    hilbert_idx: cython.int,
    a: cython.double,
    b: cython.double,
    adjusted_prev_period: cython.double,
):
    hilbert_variable.do_transform(input_value, hilbert_idx, a, b, adjusted_prev_period, False)

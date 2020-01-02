# -*- coding: utf-8 -*-

__all__ = ["Zero", "Constant"]

import theano.tensor as tt


class Zero:
    def __call__(self, x):
        return tt.zeros_like(x)


class Constant:
    def __init__(self, value):
        self.value = tt.as_tensor_variable(value)

    def __call__(self, x):
        return tt.zeros_like(x) + self.value

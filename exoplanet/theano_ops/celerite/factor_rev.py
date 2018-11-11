# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FactorRevOp"]

from .base_op import CeleriteBaseOp


class FactorRevOp(CeleriteBaseOp):

    func_file = "./factor_rev.cc"
    func_name = "APPLY_SPECIFIC(factor_rev)"
    num_input = 7
    output_ndim = (1, 2, 2, 2)

    def __init__(self, J=-1):
        super(FactorRevOp, self).__init__(J=J)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["SolveRevOp"]

from .base_op import CeleriteBaseOp


class SolveRevOp(CeleriteBaseOp):

    func_file = "./solve_rev.cc"
    func_name = "APPLY_SPECIFIC(solve_rev)"
    num_input = 8
    output_ndim = (2, 2, 1, 2, 2)

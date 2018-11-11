# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LimbDarkRevOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp


class LimbDarkRevOp(StarryBaseOp):

    __props__ = ()
    func_file = "./limbdark_rev.cc"
    func_name = "APPLY_SPECIFIC(limbdark_rev)"
    num_input = 4

    def make_node(self, *args):
        if len(args) != self.num_input:
            raise ValueError("expected {0} inputs".format(self.num_input))
        dtype = theano.config.floatX
        in_args = []
        for a in args:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [a.type() for a in in_args[:3]]
        return gof.Apply(self, in_args, out_args)

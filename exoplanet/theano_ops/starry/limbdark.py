# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LimbDarkOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp
from .limbdark_rev import LimbDarkRevOp


class LimbDarkOp(StarryBaseOp):

    __props__ = ()
    func_file = "./limbdark.cc"
    func_name = "APPLY_SPECIFIC(limbdark)"
    num_input = 3

    def __init__(self):
        self.grad_op = LimbDarkRevOp()
        super(LimbDarkOp, self).__init__()

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
        out_args = [in_args[1].type()]
        return gof.Apply(self, in_args, out_args)

    def grad(self, inputs, gradients):
        u, b, r = inputs
        bf, = gradients
        if isinstance(bf.type, theano.gradient.DisconnectedType):
            bf = tt.zeros_like(b)
        return self.grad_op(u, b, r, bf)

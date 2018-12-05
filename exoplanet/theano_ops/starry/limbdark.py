# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LimbDarkOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp


class LimbDarkOp(StarryBaseOp):

    __props__ = ()
    func_file = "./limbdark.cc"
    func_name = "APPLY_SPECIFIC(limbdark)"

    def make_node(self, c, b, r, los):
        in_args = []
        dtype = theano.config.floatX
        for a in [c, b, r, los]:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)

        out_args = [
            in_args[1].type(),
            tt.TensorType(dtype=dtype,
                          broadcastable=[False] * (in_args[1].ndim + 1))(),
            in_args[1].type(),
            in_args[2].type(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return (
            shapes[1], list(shapes[0]) + list(shapes[1]),
            shapes[1], shapes[2])

    def grad(self, inputs, gradients):
        c, b, r, los = inputs
        f, dfdcl, dfdb, dfdr = self(*inputs)
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError("can't propagate gradients wrt parameter {0}"
                                 .format(i+1))
        bc = tt.sum(tt.reshape(bf, (1, bf.size)) *
                    tt.reshape(dfdcl, (c.size, bf.size)), axis=-1)
        bb = bf * dfdb
        br = bf * dfdr
        return bc, bb, br, tt.zeros_like(los)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

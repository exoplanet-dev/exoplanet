# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["IntegratedLimbDarkOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp


class IntegratedLimbDarkOp(StarryBaseOp):

    params_type = gof.ParamsType(
        tol=theano.scalar.float64,
        min_depth=theano.scalar.int32,
        max_depth=theano.scalar.int32,
        Nc=theano.scalar.int32,
        include_contacts=theano.scalar.bool,
    )

    __props__ = ()
    func_file = "./integrated_limbdark.cc"
    func_name = "APPLY_SPECIFIC(integrated_limbdark)"

    def __init__(self, tol=1e-6, min_depth=0, max_depth=50, Nc=-1,
                 include_contacts=False, **kwargs):
        self.tol = float(tol)
        self.min_depth = max(0, int(min_depth))
        self.max_depth = max(self.min_depth + 1, int(max_depth))
        self.Nc = int(Nc)
        self.include_contacts = bool(include_contacts)
        super(IntegratedLimbDarkOp, self).__init__()

    def make_node(self, *args):
        if len(args) != 11:
            raise ValueError("wrong number of inputs")
        dtype = theano.config.floatX
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [
            in_args[1].type(),
            tt.TensorType(dtype=theano.config.floatX,
                          broadcastable=[False] * (in_args[1].ndim + 1))(),
            in_args[1].type(),
            in_args[2].type(),
            in_args[3].type(),
            in_args[4].type(),
            in_args[5].type(),
            in_args[6].type(),
            in_args[7].type(),
            tt.lscalar().type(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return (
            shapes[1], list(shapes[0]) + list(shapes[1]),
            shapes[1], shapes[2], shapes[3], shapes[4], shapes[5],
            shapes[6], shapes[7], ())

    def grad(self, inputs, gradients):
        c, r, x, xt, xtt, y, yt, ytt, z, zt, dt = inputs
        f, dfdcl, dfdr, dfdx, dfdxt, dfdxtt, dfdy, dfdyt, dfdytt, neval \
            = self(*inputs)
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError("can't propagate gradients wrt parameter {0}"
                                 .format(i+1))
        bc = tt.sum(tt.reshape(bf, (1, bf.size)) *
                    tt.reshape(dfdcl, (c.size, bf.size)), axis=-1)
        br = bf * dfdr
        bx = bf * dfdx
        bxt = bf * dfdxt
        bxtt = bf * dfdxtt
        by = bf * dfdy
        byt = bf * dfdyt
        bytt = bf * dfdytt
        return (
            bc, br, bx, bxt, bxtt, by, byt, bytt,
            tt.zeros_like(z), tt.zeros_like(zt), tt.zeros_like(dt))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

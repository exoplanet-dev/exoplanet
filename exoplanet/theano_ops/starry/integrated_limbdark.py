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

    def __init__(
        self,
        tol=1e-6,
        min_depth=0,
        max_depth=50,
        Nc=-1,
        include_contacts=False,
        **kwargs
    ):
        self.tol = float(tol)
        self.min_depth = max(0, int(min_depth))
        self.max_depth = max(self.min_depth + 1, int(max_depth))
        self.Nc = int(Nc)
        self.include_contacts = bool(include_contacts)
        super(IntegratedLimbDarkOp, self).__init__()

    def make_node(self, *args):
        if len(args) != 11:
            raise ValueError("wrong number of inputs")
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [
            in_args[1].type(),
            tt.TensorType(
                dtype=theano.config.floatX,
                broadcastable=[False] * (in_args[0].ndim + in_args[1].ndim),
            )(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            in_args[1].type(),
            tt.lscalar().type(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        shape = shapes[1]
        return (
            shape,
            list(shapes[0]) + list(shapes[1]),
            shape,
            shape,
            shape,
            shape,
            shape,
            shape,
            shape,
            shape,
            (),
        )

    def c_compile_args(self, compiler):
        args = super(IntegratedLimbDarkOp, self).c_compile_args(compiler)
        args.append("-DLIMBDARK_NC={0}".format(self.Nc))
        return args

    def grad(self, inputs, gradients):
        c = inputs[0]
        f, dcl, dr, dt, dn, daome2, de, dsinw, dcosw, dcosi, neval = self(
            *inputs
        )
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        bc = tt.sum(
            tt.reshape(bf, (1, bf.size)) * tt.reshape(dcl, (c.size, bf.size)),
            axis=-1,
        )
        return (
            bc,
            bf * dr,
            bf * dt,
            bf * dn,
            bf * daome2,
            bf * de,
            bf * dsinw,
            bf * dcosw,
            tt.zeros_like(dcosi),
            bf * dcosi,
            tt.zeros_like(bf),
        )

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

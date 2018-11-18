# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerOp"]

import theano
from theano import gof
import theano.tensor as tt


class KeplerOp(gof.COp):
    params_type = gof.ParamsType(
        maxiter=theano.scalar.int64,
        tol=theano.scalar.float64,
    )
    __props__ = ("tol", "maxiter")
    func_file = "./solver.cc"
    func_name = "APPLY_SPECIFIC(solver)"

    def __init__(self, tol=1e-8, maxiter=2000, **kwargs):
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        super(KeplerOp, self).__init__(self.func_file, self.func_name)

    def make_node(self, mean_anom, eccen):
        in_args = [tt.as_tensor_variable(mean_anom),
                   tt.as_tensor_variable(eccen)]
        return gof.Apply(self, in_args, [in_args[0].type()])

    def infer_shape(self, node, shapes):
        return shapes[0],

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def grad(self, inputs, gradients):
        M, e = inputs
        E = self(M, e)
        bM = gradients[0] / (1.0 - e * tt.cos(E))
        be = tt.sin(E) * bM
        return [bM, be]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

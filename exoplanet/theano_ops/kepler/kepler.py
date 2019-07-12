# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerOp"]

import theano
from theano import gof
import theano.tensor as tt

from ..build_utils import get_cache_version, get_compile_args, get_header_dirs


class KeplerOp(gof.COp):
    __props__ = ()
    func_file = "./kepler.cc"
    func_name = "APPLY_SPECIFIC(kepler)"

    def __init__(self, **kwargs):
        super(KeplerOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_headers(self, compiler):
        return [
            "exoplanet/theano_helpers.h",
            "exoplanet/kepler.h"
        ]

    def c_header_dirs(self, compiler):
        return get_header_dirs(eigen=False)

    def c_compile_args(self, compiler):
        return get_compile_args(compiler)

    def make_node(self, mean_anom, eccen):
        in_args = [tt.as_tensor_variable(mean_anom),
                   tt.as_tensor_variable(eccen)]
        return gof.Apply(self, in_args,
                         [in_args[0].type(), in_args[0].type(),
                          in_args[0].type(), in_args[0].type()])

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0], shapes[0]

    def grad(self, inputs, gradients):
        M, e = inputs
        sinE, cosE, sinf, cosf = self(M, e)

        bM = tt.zeros_like(M)
        be = tt.zeros_like(M)

        # Pre-define the E derivatives
        ecosE = e * cosE
        dEdM = 1 / (1 - ecosE)
        dEde = sinE * dEdM

        # Pre-define the f derivatives
        sqrt1me2 = tt.sqrt(1 - e**2)
        omecosE = 1 - e*cosE
        dfdE = sqrt1me2 / omecosE
        dfde = sinE / (sqrt1me2 * omecosE)

        if not isinstance(gradients[0].type, theano.gradient.DisconnectedType):
            bM += gradients[0] * cosE * dEdM
            be += gradients[0] * cosE * dEde

        if not isinstance(gradients[1].type, theano.gradient.DisconnectedType):
            bM -= gradients[1] * sinE * dEdM
            be -= gradients[1] * sinE * dEde

        if not isinstance(gradients[2].type, theano.gradient.DisconnectedType):
            inner = cosf * dfdE
            bM += gradients[2] * inner * dEdM
            be += gradients[2] * (inner * dEde + cosf * dfde)

        if not isinstance(gradients[3].type, theano.gradient.DisconnectedType):
            inner = sinf * dfdE
            bM -= gradients[3] * inner * dEdM
            be -= gradients[3] * (inner * dEde + sinf * dfde)

        return [bM, be]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

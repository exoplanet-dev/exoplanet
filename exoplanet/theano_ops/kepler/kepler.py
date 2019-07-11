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
                          in_args[0].type()])

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0]

    def grad(self, inputs, gradients):
        M, e = inputs
        E, sinf, cosf = self(M, e)

        bM = tt.zeros_like(M)
        be = tt.zeros_like(M)
        ecosE = e * tt.cos(E)

        if not isinstance(gradients[0].type, theano.gradient.DisconnectedType):
            # Backpropagate E_bar
            bM = gradients[0] / (1 - ecosE)
            be = tt.sin(E) * bM

        bsinf = gradients[1]
        fs = isinstance(bsinf.type, theano.gradient.DisconnectedType)
        bcosf = gradients[2]
        fc = isinstance(bcosf.type, theano.gradient.DisconnectedType)

        if not (fs and fc):
            bf = tt.zeros_like(M)
            if not fs:
                bf += tt.as_tensor_variable(bsinf) * cosf
            if not fc:
                bf -= tt.as_tensor_variable(bcosf) * sinf

            # Backpropagate f_bar
            tanf2 = sinf / (1 + cosf)  # tan(0.5*f)
            e2 = e**2
            ome2 = 1 - e2
            ome = 1 - e
            ope = 1 + e
            # cosf22 = cosf2**2
            cosf22 = 0.5*(1 + cosf)
            twoecosf22 = 2 * e * cosf22
            factor = tt.sqrt(ope/ome)
            inner = (twoecosf22+ome) * bf

            bM += factor*(ome*tanf2**2+ope)*inner*cosf22/(ope*ome2)
            be += -2*cosf22*tanf2/ome2**2*inner*(ecosE-2+e2)

        return [bM, be]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

# -*- coding: utf-8 -*-

__all__ = ["kepler"]

import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano

from ..utils import as_tensor_variable
from . import driver
from .compat import Apply, Op
from .helpers import resize_or_set


class Kepler(Op):
    __props__ = ()

    def make_node(self, M, ecc):
        in_args = [as_tensor_variable(M), as_tensor_variable(ecc)]
        if any(i.dtype != "float64" for i in in_args):
            raise ValueError("float64 dtypes are required for Kepler op")
        out_args = [in_args[0].type(), in_args[1].type()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return args[-1]

    def perform(self, node, inputs, outputs):
        M, ecc = inputs
        sinf = resize_or_set(outputs, 0, M.shape)
        cosf = resize_or_set(outputs, 1, M.shape)
        driver.kepler(M % (2 * np.pi), ecc, sinf, cosf)

    def grad(self, inputs, gradients):
        M, e = inputs
        sinf, cosf = self(M, e)

        bM = tt.zeros_like(M)
        be = tt.zeros_like(M)

        # e * cos(f)
        ecosf = e * cosf

        # 1 - e^2
        ome2 = 1 - e ** 2

        # Partials
        dfdM = (1 + ecosf) ** 2 / ome2 ** 1.5
        dfde = (2 + ecosf) * sinf / ome2

        if not isinstance(gradients[0].type, theano.gradient.DisconnectedType):
            bM += gradients[0] * cosf * dfdM
            be += gradients[0] * cosf * dfde

        if not isinstance(gradients[1].type, theano.gradient.DisconnectedType):
            bM -= gradients[1] * sinf * dfdM
            be -= gradients[1] * sinf * dfde

        return [bM, be]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


kepler = Kepler()

# -*- coding: utf-8 -*-

__all__ = ["KeplerOp"]

import theano
import theano.tensor as tt
from theano import gof

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
        return ["exoplanet/theano_helpers.h", "exoplanet/kepler.h"]

    def c_header_dirs(self, compiler):
        return get_header_dirs(eigen=False)

    def c_compile_args(self, compiler):
        return get_compile_args(compiler)

    def make_node(self, mean_anom, eccen):
        in_args = [
            tt.as_tensor_variable(mean_anom),
            tt.as_tensor_variable(eccen),
        ]
        return gof.Apply(self, in_args, [in_args[0].type(), in_args[0].type()])

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0]

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

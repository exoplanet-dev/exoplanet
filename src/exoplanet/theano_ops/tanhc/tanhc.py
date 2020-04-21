# -*- coding: utf-8 -*-

__all__ = ["tanhc", "atanhc"]

import numpy as np
from theano.scalar.basic import (
    UnaryScalarOp,
    complex_types,
    float_types,
    upgrade_to_float,
)
from theano.tensor.elemwise import Elemwise

from ..build_utils import get_cache_version


class Tanhc(UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        if np.abs(x) < 1e-5:
            return 1 - x ** 2 / 3
        return np.tanh(x) / x

    def impl(self, x):
        return Tanhc.st_impl(x)

    def c_code_cache_version(self):
        return get_cache_version()

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        return [gz * scaler_tanhc_grad(x)]

    def c_support_code(self):
        return """
            double _tanhc(double x) {
                if (x < -1e-5 || 1e-5 < x) return tanh(x) / x;
                return 1 - x * x / 3;
            }
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return (
                """%(z)s =
                _tanhc(%(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floating point is implemented")


class TanhcGrad(UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        if np.abs(x) < 1e-5:
            return -2 * x / 3.0
        tanh = np.tanh(x)
        return (1 - tanh ** 2) / x - tanh / x ** 2

    def impl(self, x):
        return Tanhc.st_impl(x)

    def c_code_cache_version(self):
        return get_cache_version()

    def L_op(self, inputs, outputs, grads):
        raise NotImplementedError()

    def c_support_code(self):
        return """
            double _tanhc_grad(double x) {
                if (x < -1e-5 || 1e-5 < x) {
                    double th = tanh(x);
                    return (1 - th * th) / x - th / (x * x);
                }
                return -2.0 * x / 3.0;
            }
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return (
                """%(z)s =
                _tanhc_grad(%(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floating point is implemented")


class Atanhc(UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        if np.abs(x) < 1e-5:
            return 1 + x ** 2 / 3
        return np.arctanh(x) / x

    def impl(self, x):
        return Atanhc.st_impl(x)

    def c_code_cache_version(self):
        return get_cache_version()

    def L_op(self, inputs, outputs, grads):
        # FIXME: still nan at zero
        (x,) = inputs
        (atanhc,) = outputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        return [gz * ((x * (1 - x ** 2)) ** -1 - atanhc * x ** -1)]

    def c_support_code(self):
        return """
            double _atanhc(double x) {
                if (x < -1e-5 || 1e-5 < x) return atanh(x) / x;
                return 1 + x * x / 3;
            }
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return (
                """%(z)s =
                _atanhc(%(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floating point is implemented")


scaler_tanhc = Tanhc(upgrade_to_float, name="tanhc")
tanhc = Elemwise(scaler_tanhc, name="Elemwise{tanhc}", nfunc_spec=None)
scaler_tanhc_grad = TanhcGrad(upgrade_to_float, name="tanhc_grad")
scaler_atanhc = Atanhc(upgrade_to_float, name="atanhc")
atanhc = Elemwise(scaler_atanhc, name="Elemwise{atanhc}", nfunc_spec=None)

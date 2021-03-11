# -*- coding: utf-8 -*-

import aesara_theano_fallback.tensor as tt
import numpy as np
import pytest
from aesara_theano_fallback import aesara as theano

from exoplanet.theano_ops.rebound import ReboundOp
from exoplanet.theano_ops.test_tools import InferShapeTester

try:
    import rebound  # NOQA
except ImportError:
    pytest.skip(
        "skipping rebound tests when not installed", allow_module_level=True
    )


@pytest.mark.filterwarnings("ignore:For better performance")
class TestRebound(InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = ReboundOp
        self.op = ReboundOp()

    def get_args(self):
        m_val = np.array([1.3, 1e-3, 1e-5])
        x_val = np.zeros((3, 6))
        x_val[1, 0] = 15.0
        x_val[1, 4] = 0.4
        x_val[2, 0] = 100.0
        x_val[2, 4] = 0.2
        t = np.linspace(100, 1000, 12)

        m = tt.dvector()
        x = tt.dmatrix()

        f = theano.function([m, x], self.op(m, x, t)[0])

        return t, f, [m, x], [m_val, x_val]

    def test_basic(self):
        _, f, _, in_args = self.get_args()
        f(*in_args)

    def test_infer_shape(self):
        t, f, args, arg_vals = self.get_args()
        self._compile_and_check(
            args, self.op(*(list(args) + [t])), arg_vals, self.op_class
        )

    def test_grad(self):
        t, _, _, in_args = self.get_args()
        func = lambda *args: self.op(*(list(args) + [t]))[0]  # NOQA
        theano.gradient.verify_grad(func, in_args, n_tests=1, rng=np.random)

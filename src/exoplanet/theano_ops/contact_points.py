# -*- coding: utf-8 -*-

__all__ = ["contact_points"]

import aesara_theano_fallback.tensor as tt
import numpy as np

from ..utils import as_tensor_variable
from . import driver
from .compat import Apply, Op
from .helpers import resize_or_set


class ContactPoints(Op):
    __props__ = ()

    def __init__(self, tol=1e-10):
        self.tol = float(tol)
        super().__init__()

    def make_node(self, *inputs):
        in_args = [as_tensor_variable(i) for i in inputs]
        if any(i.dtype != "float64" for i in in_args):
            raise ValueError(
                "float64 dtypes are required for ContactPoints op; "
                "got:\n{0}".format([i.dtype for i in inputs])
            )
        out_args = [
            in_args[0].type(),
            in_args[0].type(),
            tt.TensorType(
                dtype="int32", broadcastable=[False] * in_args[0].ndim
            )(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[0], shapes[0], shapes[0]

    def perform(self, node, inputs, outputs):
        a, e, cosw, sinw, cosi, sini, L = inputs
        M_left = resize_or_set(outputs, 0, a.shape)
        M_right = resize_or_set(outputs, 1, a.shape)
        flag = resize_or_set(outputs, 2, a.shape, dtype=np.int32)
        driver.contact_points(
            a, e, cosw, sinw, cosi, sini, L, M_left, M_right, flag, self.tol
        )


contact_points = ContactPoints()

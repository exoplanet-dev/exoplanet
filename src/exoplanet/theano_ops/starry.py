# -*- coding: utf-8 -*-

__all__ = ["get_cl", "limbdark"]

import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano
from scipy.optimize import bisect

from ..utils import as_tensor_variable
from . import driver
from .compat import Apply, Op
from .helpers import resize_or_set


class GetClRev(Op):
    __props__ = ()
    itypes = (tt.dvector,)
    otypes = (tt.dvector,)

    def infer_shape(self, *args):
        return args[-1]

    def perform(self, node, inputs, outputs):
        bc = inputs[0]
        bu = resize_or_set(outputs, 0, bc.shape)
        driver.get_cl_rev(bc, bu)


get_cl_rev = GetClRev()


class GetCl(Op):

    __props__ = ()
    itypes = (tt.dvector,)
    otypes = (tt.dvector,)

    def infer_shape(self, *args):
        return args[-1]

    def perform(self, node, inputs, outputs):
        u = inputs[0]
        c = resize_or_set(outputs, 0, u.shape)
        driver.get_cl(u, c)

    def grad(self, inputs, gradients):
        return (get_cl_rev(gradients[0]),)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


get_cl = GetCl()


class LimbDark(Op):
    __props__ = ()

    def __init__(self):
        self.ld = driver.LimbDark()
        super().__init__()

    def __getstate__(self):
        return {}

    def __setstate__(self, data):
        self.ld = driver.LimbDark()

    def make_node(self, *inputs):
        in_args = [as_tensor_variable(i) for i in inputs]
        if any(i.dtype != "float64" for i in in_args):
            raise ValueError(
                "float64 dtypes are required for LimbDark op; "
                "got:\n{0}".format([i.dtype for i in inputs])
            )
        out_args = [
            in_args[1].type(),
            tt.TensorType(
                dtype="float64", broadcastable=[False] * (in_args[1].ndim + 1)
            )(),
            in_args[1].type(),
            in_args[2].type(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return (
            shapes[1],
            list(shapes[0]) + list(shapes[1]),
            shapes[1],
            shapes[2],
        )

    def perform(self, node, inputs, outputs):
        cl, b, r, los = inputs
        f = resize_or_set(outputs, 0, b.shape)
        dfdcl = resize_or_set(outputs, 1, cl.shape + b.shape)
        dfdb = resize_or_set(outputs, 2, b.shape)
        dfdr = resize_or_set(outputs, 3, b.shape)
        self.ld.apply(cl, b, r, los, f, dfdcl, dfdb, dfdr)

    def grad(self, inputs, gradients):
        c, b, r, los = inputs
        f, dfdcl, dfdb, dfdr = self(*inputs)
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        bc = tt.sum(
            tt.reshape(bf, (1, bf.size))
            * tt.reshape(dfdcl, (c.size, bf.size)),
            axis=-1,
        )
        bb = bf * dfdb
        br = bf * dfdr
        return bc, bb, br, tt.zeros_like(los)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


limbdark = LimbDark()


#
# Helpers for Radius From Occ Area
#


def kite_area(r, b):
    abc = tt.sort(tt.stack((r, b, tt.ones_like(r)), axis=0), axis=0)
    A = abc[0]
    B = abc[1]
    C = abc[2]
    return 0.5 * tt.sqrt(
        tt.abs_((A + (B + C)) * (C - (A - B)) * (C + (A - B)) * (A + (B - C)))
    )


def depth(r, b):
    # depth = 1 - s0 / pi; where s0 is from Agol+
    b = tt.abs_(b)
    r = tt.abs_(r)
    b2 = b ** 2
    r2 = r ** 2
    opr = 1 + r
    omr = 1 - r
    rmo = r - 1

    # Case 2
    a = kite_area(r, b)
    twoa = 2 * a
    k0 = tt.arctan2(twoa, rmo * opr + b2)
    k1 = tt.arctan2(twoa, omr * opr + b2)
    case2 = (k1 + r2 * k0 - a) / np.pi

    return tt.switch(
        tt.le(opr, b),
        tt.zeros_like(r),
        tt.switch(
            tt.and_(tt.lt(tt.abs_(omr), b), tt.lt(b, opr)),
            case2,
            tt.switch(tt.le(b, omr), r2, tt.ones_like(r)),
        ),
    )


def depth_grad(r, b):
    # depth = 1 - s0 / pi; where s0 is from Agol+
    b = tt.abs_(b)
    r = tt.abs_(r)
    b2 = b ** 2
    opr = 1 + r
    omr = 1 - r
    rmo = r - 1

    # Case 2
    a = kite_area(r, b)
    twor = 2 * r
    twoa = 2 * a
    k0 = tt.arctan2(twoa, rmo * opr + b2)
    dr = twor * k0 / np.pi
    db = -twoa / (b * np.pi)

    zero = tt.zeros_like(r)
    return (
        tt.switch(
            tt.le(opr, b),
            zero,
            tt.switch(
                tt.and_(tt.lt(tt.abs_(omr), b), tt.lt(b, opr)),
                dr,
                tt.switch(tt.le(b, omr), twor, zero),
            ),
        ),
        tt.switch(
            tt.le(opr, b),
            zero,
            tt.switch(
                tt.and_(tt.lt(tt.abs_(omr), b), tt.lt(b, opr)), db, zero
            ),
        ),
    )


#
# /helpers
#


class RadiusFromOccArea(Op):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Set up the problem using Theano ops
        b = tt.dscalar()
        delta = tt.dscalar()
        r = tt.dscalar()

        # We'll need test vals to keep PyMC3 happy
        b.tag.test_value = 0.0
        delta.tag.test_value = 0.0
        r.tag.test_value = 0.0

        # Compile the objective function (this is all a little meta)
        value = delta - depth(r, b)
        self.func = theano.function([r, b, delta], value)

        super().__init__()

    def make_node(self, delta, b):
        in_args = [as_tensor_variable(delta), as_tensor_variable(b)]
        out_args = [in_args[1].type()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return (shapes[1],)

    def perform(self, node, inputs, outputs):
        delta, b = inputs
        shape = np.shape(b)

        b = np.atleast_1d(b)
        delta = np.atleast_1d(delta)
        if b.ndim != 1 or b.shape != delta.shape:
            raise ValueError("dimension mismatch")

        if np.any(b < 0) or np.any(delta < 0) or np.any(delta > 1):
            raise ValueError(
                "invalid input; valid ranges 0 <= delta <= 1; 0 <= b"
            )

        # Compute the bounds
        rmin = np.maximum(b - 1, 0)
        rmax = b + 1

        # Solve for each entry
        r = np.zeros_like(b)
        for n in range(len(b)):
            args = (b[n], delta[n])
            r[n] = bisect(
                self.func, rmin[n], rmax[n], args=args, **self.kwargs
            )

        outputs[0][0] = r.reshape(shape)

    def grad(self, inputs, gradients):
        delta, b = inputs
        r = self(*inputs)
        br = gradients[0]

        dr, db = depth_grad(r, b)
        bdelta = br / dr
        bb = -db * bdelta

        return bdelta, bb

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

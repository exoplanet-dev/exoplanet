# -*- coding: utf-8 -*-

__all__ = ["RadiusFromOccAreaOp"]

import numpy as np
import theano
import theano.tensor as tt
from scipy.optimize import bisect
from theano import gof


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


class RadiusFromOccAreaOp(gof.Op):
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

        super(RadiusFromOccAreaOp, self).__init__()

    def make_node(self, delta, b):
        in_args = [tt.as_tensor_variable(delta), tt.as_tensor_variable(b)]
        out_args = [in_args[1].type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
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

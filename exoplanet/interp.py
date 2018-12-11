# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RegularGridInterpolator"]

import theano.tensor as tt
from .theano_ops.interp import RegularGridOp


class RegularGridInterpolator(object):
    """Linear interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a filled regular grid, but the spacing may be
    uneven in any of the dimensions.

    Args:
        points: A list of vectors with shapes ``(m1,), ... (mn,)``. These
            define the grid points in each dimension.
        values: A tensor defining the values at each point in the grid
            defined by ``points``. This must have the shape
            ``(m1, ... mn, ..., nout)``.
        xi: A matrix defining the coordinates where the interpolation
            should be evaluated. This must have the shape ``(ntest, ndim)``.
        check_sorted: If ``True`` (default), check that the tensors in
            ``points`` are all sorted in ascending order. This can be set to
            ``False`` if the axes are known to be sorted, but the results will
            be unpredictable if this ends up being wrong.
        bounds_error: If ``False`` (default) extrapolate beyond the edges of
            the grid. Otherwise raise an exception.
        nout: An integer indicating the number of outputs if known at compile
            time. The default is to allow any number of outputs, but
            performance can be better if this is provided.

    """

    def __init__(self, points, values, check_sorted=True, bounds_error=False,
                 nout=-1):
        self.ndim = len(points)
        self.nout = int(nout)

        self.points = [tt.as_tensor_variable(p) for p in points]
        self.values = tt.as_tensor_variable(values)
        if self.values.ndim == self.ndim:
            self.values = tt.shape_padright(self.values)

        self.check_sorted = bool(check_sorted)
        self.bounds_error = bool(bounds_error)

        self.interp_op = RegularGridOp(self.ndim, nout=self.nout,
                                       check_sorted=self.check_sorted,
                                       bounds_error=self.bounds_error)

    def evaluate(self, t):
        return self.interp_op(t, self.values, *self.points)[0]

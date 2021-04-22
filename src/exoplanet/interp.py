# -*- coding: utf-8 -*-

__all__ = ["regular_grid_interp", "RegularGridInterpolator"]

import itertools

import aesara_theano_fallback.tensor as tt

from .utils import as_tensor_variable


def regular_grid_interp(points, values, coords):
    """Perform a linear interpolation in N-dimensions w a regular grid

    The data must be defined on a filled regular grid, but the spacing may be
    uneven in any of the dimensions.

    This implementation is based on the implementation in the
    ``scipy.interpolate.RegularGridInterpolator`` class which, in turn, is
    based on the implementation from Johannes Buchner's ``regulargrid``
    package https://github.com/JohannesBuchner/regulargrid.


    Args:
        points: A list of vectors with shapes ``(m1,), ... (mn,)``. These
            define the grid points in each dimension.
        values: A tensor defining the values at each point in the grid
            defined by ``points``. This must have the shape
            ``(m1, ... mn, ..., nout)``.
        coords: A matrix defining the coordinates where the interpolation
            should be evaluated. This must have the shape ``(ntest, ndim)``.
    """
    points = [as_tensor_variable(p) for p in points]
    ndim = len(points)
    values = as_tensor_variable(values)
    coords = as_tensor_variable(coords)

    # Find where the points should be inserted
    indices = []
    norm_distances = []
    for n, grid in enumerate(points):
        x = coords[..., n]
        i = tt.extra_ops.searchsorted(grid, x) - 1
        i = tt.clip(i, 0, grid.shape[0] - 2)
        indices.append(i)
        norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))

    result = tt.zeros(tuple(coords.shape[:-1]) + tuple(values.shape[ndim:]))
    for edge_indices in itertools.product(*((i, i + 1) for i in indices)):
        weight = tt.ones(coords.shape[:-1])
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= tt.where(tt.eq(ei, i), 1 - yi, yi)
        result += values[edge_indices] * weight
    return result


class RegularGridInterpolator:
    """Linear interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a filled regular grid, but the spacing may be
    uneven in any of the dimensions.

    Args:
        points: A list of vectors with shapes ``(m1,), ... (mn,)``. These
            define the grid points in each dimension.
        values: A tensor defining the values at each point in the grid
            defined by ``points``. This must have the shape
            ``(m1, ... mn, ..., nout)``.
    """

    def __init__(self, points, values, **kwargs):
        self.ndim = len(points)
        self.points = points
        self.values = values

    def evaluate(self, t):
        """Interpolate the data

        Args:
            t: A matrix defining the coordinates where the interpolation
                should be evaluated. This must have the shape
                ``(ntest, ndim)``.
        """
        return regular_grid_interp(self.points, self.values, t)

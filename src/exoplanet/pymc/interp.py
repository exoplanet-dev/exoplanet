# -*- coding: utf-8 -*-

__all__ = ["RegularGridInterpolator"]

import itertools

from . import compat
from .compat import numpy as np


def _ndim_coords_from_arrays(points, ndim=None):
    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError(
                    "coordinate arrays do not have the same shape"
                )
        points = np.empty(p[0].shape + (len(points),))
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        points = compat.as_tensor(points)
        if points.ndim == 1:
            if ndim is None:
                points = np.reshape(points, (-1, 1))
            else:
                points = np.reshape(points, (-1, ndim))
    return points


class RegularGridInterpolator:
    def __init__(
        self,
        points,
        values,
        method="linear",
        bounds_error=True,
        # fill_value=np.nan,
    ):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        values = compat.as_tensor(values)
        if len(points) > values.ndim:
            raise ValueError(
                f"There are {len(points)} point arrays, but values has "
                f"{values.ndim} dimensions"
            )

        for i, p in enumerate(points):
            if not p.ndim == 1:
                raise ValueError(
                    f"The points in dimension {i} must be 1-dimensional"
                )
        self.grid = tuple([compat.as_tensor(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        xi = compat.as_tensor(xi)
        indices, norm_distances = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances)

        return result

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(compat.eq(ei, i), 1 - yi, yi)
            values += self.values[edge_indices] * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = [
            np.where(yi <= 0.5, i, i + 1)
            for i, yi in zip(indices, norm_distances)
        ]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # iterate through dimensions
        for n, grid in enumerate(self.grid):
            x = xi[n]
            i = compat.searchsorted(grid, x) - 1
            i = compat.set_subtensor(i < 0, i, 0)
            i = compat.set_subtensor(i > grid.size - 2, i, grid.size - 2)
            indices.append(i)
            norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
        return indices, norm_distances

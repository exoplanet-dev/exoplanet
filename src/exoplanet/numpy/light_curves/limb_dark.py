# -*- coding: utf-8 -*-

__all__ = ["LimbDarkLightCurve"]

import numpy
from numpy import pi

from .. import compat
from ..citations import add_citations_to_model
from ..compat import numpy as np


class LimbDarkLightCurve:
    """A quadratically limb darkened light curve

    Args:
        u1 (scalar): The first limb darkening coefficient.
        u2 (scalar): The second limb darkening coefficient.

    """

    __citations__ = ("starry",)

    def __init__(self, u1, u2, *, model=None):
        add_citations_to_model(self.__citations__, model=model)
        if not compat.isscalar(u1):
            raise ValueError(
                "Since v0.5, exoplanet only supports quadratic limb darkening "
                "and the parameters must be provided as scalars"
            )
        self.u1 = u1
        self.u2 = u2
        self.c = compat.as_tensor([1 - u1 - 1.5 * u2, u1 + 2 * u2, -0.25 * u2])
        self.c /= pi * (self.c[0] + self.c[1] / 1.5)

    def get_ror_from_approx_transit_depth(self, delta, b, jac=False):
        """Get the radius ratio corresponding to a particular transit depth

        This result will be approximate and it requires ``|b| < 1`` because it
        relies on the small planet approximation.

        Args:
            delta (tensor): The approximate transit depth in relative units
            b (tensor): The impact parameter
            jac (bool): If true, the Jacobian ``d ror / d delta`` is also
                returned

        Returns:
            ror: The radius ratio that approximately corresponds to the depth
            ``delta`` at impact parameter ``b``.

        """
        b = compat.as_tensor(b)
        delta = compat.as_tensor(delta)
        f0 = 1 - 2 * (self.u1 / 6.0 + self.u2 / 12.0)
        arg = 1 - np.sqrt(1 - b ** 2)
        f = 1 - (self.u1 * arg + self.u2 * arg ** 2)
        factor = f0 / f
        ror = np.sqrt(delta * factor)
        if not jac:
            return np.reshape(ror, b.shape)
        drorddelta = 0.5 * factor / ror
        return np.reshape(ror, b.shape), np.reshape(drorddelta, b.shape)

    def get_light_curve(
        self,
        *,
        orbit,
        t,
        texp=None,
        oversample=7,
        order=0,
        use_in_transit=None,
        light_delay=False,
    ):
        """Get the light curve for an orbit at a set of times

        Args:
            orbit: An object with a ``get_relative_position`` method that
                takes a tensor of times and returns a list of Cartesian
                coordinates of a set of bodies relative to the central source.
                This method should return three tensors (one for each
                coordinate dimension) and each tensor should have the shape
                ``append(t.shape, r.shape)`` or ``append(t.shape, oversample,
                r.shape)`` when ``texp`` is given. The first two coordinate
                dimensions are treated as being in the plane of the sky and the
                third coordinate is the line of sight with positive values
                pointing *away* from the observer. For an example, take a look
                at :class:`orbits.KeplerianOrbit`.
            t (tensor): The times where the light curve should be evaluated.
            texp (Optional[tensor]): The exposure time of each observation.
                This can be a scalar or a tensor with the same shape as ``t``.
                If ``texp`` is provided, ``t`` is assumed to indicate the
                timestamp at the *middle* of an exposure of length ``texp``.
            oversample (Optional[int]): The number of function evaluations to
                use when numerically integrating the exposure time.
            order (Optional[int]): The order of the numerical integration
                scheme. This must be one of the following: ``0`` for a
                centered Riemann sum (equivalent to the "resampling" procedure
                suggested by Kipping 2010), ``1`` for the trapezoid rule, or
                ``2`` for Simpson's rule.
            use_in_transit (Optional[bool]): If ``True``, the model will only
                be evaluated for the data points expected to be in transit
                as computed using the ``in_transit`` method on ``orbit``.

        """
        use_in_transit = (
            not light_delay if use_in_transit is None else use_in_transit
        )

        r = np.reshape(orbit.r_planet, (orbit.r_planet.size,))
        t = compat.as_tensor(t)

        if use_in_transit:
            model = np.zeros_like(r) + np.zeros_like(t)[..., None]
            inds = orbit.in_transit(t, texp=texp, light_delay=light_delay)
            t = t[inds]

        if texp is None:
            tgrid = t
        else:
            texp = compat.as_tensor(texp)
            dt, stencil = get_stencil(order, oversample)
            if texp.ndim == 0:
                dt = texp * dt
            else:
                if use_in_transit:
                    dt = texp[inds][..., None] * dt
                else:
                    dt = texp[..., None] * dt
            tgrid = t[..., None] + dt

        rgrid = r + np.zeros_like(tgrid)[..., None]

        coords = orbit.get_relative_position(tgrid, light_delay=light_delay)
        b = np.sqrt(coords[0] ** 2 + coords[1] ** 2) / orbit.r_star
        b = np.reshape(b, rgrid.shape)
        mask = np.reshape(coords[2], rgrid.shape) > 0

        lc = np.zeros_like(b)
        lc = compat.set_subtensor(
            mask,
            lc,
            self._compute_light_curve(b[mask], rgrid[mask] / orbit.r_star),
        )
        if texp is not None:
            lc = np.sum(stencil[:, None] * lc, axis=-2)

        if use_in_transit:
            model = compat.set_subtensor(inds, model, lc)
            return model
        return lc

    def _compute_light_curve(self, b, r):
        """Compute the light curve for a set of impact parameters and radii

        .. note:: The stellar radius is *not* included in this method so the
            coordinates should be in units of the star's radius.

        Args:
            b (tensor): A tensor of impact parameter values.
            r (tensor): A tensor of radius ratios with the same shape as ``b``.

        """
        b = compat.as_tensor(b)
        return np.dot(compat.ops.quad_solution_vector(b, r), self.c) - 1


def get_stencil(order, oversample):
    oversample = int(oversample)
    oversample += 1 - oversample % 2
    stencil = numpy.ones(oversample)

    if order == 0:
        dt = numpy.linspace(-0.5, 0.5, 2 * oversample + 1)[1:-1:2]
    elif order == 1:
        dt = numpy.linspace(-0.5, 0.5, oversample)
        stencil[1:-1] = 2
    elif order == 2:
        dt = numpy.linspace(-0.5, 0.5, oversample)
        stencil[1:-1:2] = 4
        stencil[2:-1:2] = 2
    else:
        raise ValueError("order must be <= 2")

    stencil /= numpy.sum(stencil)

    return dt, stencil

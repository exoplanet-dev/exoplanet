# -*- coding: utf-8 -*-

__all__ = ["LimbDarkLightCurve"]

import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano
from exoplanet_core.pymc import ops

from ..citations import add_citations_to_model
from ..utils import as_tensor_variable, deprecation_warning


def get_cl(u1, u2):
    u1 = as_tensor_variable(u1)
    u2 = as_tensor_variable(u2)
    c0 = 1 - u1 - 1.5 * u2
    c1 = u1 + 2 * u2
    c2 = -0.25 * u2
    norm = np.pi * (c0 + c1 / 1.5)
    return tt.stack([c0, c1, c2]) / norm


def quad_limbdark_light_curve(c, b, r):
    b = as_tensor_variable(b)
    r = as_tensor_variable(r)
    return tt.dot(ops.quad_solution_vector(b, r), c) - 1.0


class LimbDarkLightCurve:
    """A quadratically limb darkened light curve

    .. note:: Earlier versions of exoplanet supported higher (and lower) order
        limb darkening, but this support was removed in v0.5.0. For higher
        order limb darkening profiles, use starry directly.

    Args:
        u1 (scalar): The first limb darkening coefficient
        u2 (scalar): The second limb darkening coefficient
    """

    __citations__ = ("starry",)

    def __init__(self, u1, u2=None, model=None):
        add_citations_to_model(self.__citations__, model=model)
        if u2 is None:
            deprecation_warning(
                "using a vector of limb darkening coefficients is deprecated; "
                "use u1 and u2 directly"
            )

            # If a vector is provided, we need to assert that it is 2D
            msg = (
                "only quadratic limb darkening is supported; "
                "use `starry` for more flexibility"
            )
            try:
                assert_op = theano.assert_op.Assert(msg)
            except AttributeError:
                assert_op = tt.opt.Assert(msg)
            u1 = as_tensor_variable(u1)
            u1 = assert_op(
                u1, tt.and_(tt.eq(u1.ndim, 1), tt.eq(u1.shape[0], 2))
            )

            self.u1 = u1[0]
            self.u2 = u1[1]
        else:
            self.u1 = as_tensor_variable(u1)
            self.u2 = as_tensor_variable(u2)
        self.c = get_cl(self.u1, self.u2)

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
        b = as_tensor_variable(b)
        delta = as_tensor_variable(delta)
        f0 = 1 - 2 * self.u1 / 6.0 - 2 * self.u2 / 12.0
        arg = 1 - tt.sqrt(1 - b ** 2)
        f = 1 - self.u1 * arg - self.u2 * arg ** 2
        factor = f0 / f
        ror = tt.sqrt(delta * factor)
        if not jac:
            return tt.reshape(ror, b.shape)
        drorddelta = 0.5 * factor / ror
        return tt.reshape(ror, b.shape), tt.reshape(drorddelta, b.shape)

    def get_light_curve(
        self,
        orbit=None,
        r=None,
        t=None,
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
            r (tensor): The radius of the transiting body in the same units as
                ``r_star``. This should have a shape that is consistent with
                the coordinates returned by ``orbit``. In general, this means
                that it should probably be a scalar or a vector with one entry
                for each body in ``orbit``. Note that this is a different
                quantity than the planet-to-star radius ratio; do not confuse
                the two!
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
        if orbit is None:
            raise ValueError("missing required argument 'orbit'")
        if r is None:
            raise ValueError("missing required argument 'r'")
        if t is None:
            raise ValueError("missing required argument 't'")

        use_in_transit = (
            not light_delay if use_in_transit is None else use_in_transit
        )

        r = as_tensor_variable(r)
        r = tt.reshape(r, (r.size,))
        t = as_tensor_variable(t)

        # If use_in_transit, we should only evaluate the model at times where
        # at least one planet is transiting
        if use_in_transit:
            transit_model = tt.shape_padleft(
                tt.zeros_like(r), t.ndim
            ) + tt.shape_padright(tt.zeros_like(t), r.ndim)
            inds = orbit.in_transit(t, r=r, texp=texp, light_delay=light_delay)
            t = t[inds]

        # Handle exposure time integration
        if texp is None:
            tgrid = t
            rgrid = tt.shape_padleft(r, tgrid.ndim) + tt.shape_padright(
                tt.zeros_like(tgrid), r.ndim
            )
        else:
            texp = as_tensor_variable(texp)

            oversample = int(oversample)
            oversample += 1 - oversample % 2
            stencil = np.ones(oversample)

            # Construct the exposure time integration stencil
            if order == 0:
                dt = np.linspace(-0.5, 0.5, 2 * oversample + 1)[1:-1:2]
            elif order == 1:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1] = 2
            elif order == 2:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1:2] = 4
                stencil[2:-1:2] = 2
            else:
                raise ValueError("order must be <= 2")
            stencil /= np.sum(stencil)

            if texp.ndim == 0:
                dt = texp * dt
            else:
                if use_in_transit:
                    dt = tt.shape_padright(texp[inds]) * dt
                else:
                    dt = tt.shape_padright(texp) * dt
            tgrid = tt.shape_padright(t) + dt

            # Madness to get the shapes to work out...
            rgrid = tt.shape_padleft(r, tgrid.ndim) + tt.shape_padright(
                tt.zeros_like(tgrid), 1
            )

        # Evalute the coordinates of the transiting body in the plane of the
        # sky
        coords = orbit.get_relative_position(tgrid, light_delay=light_delay)
        b = tt.sqrt(coords[0] ** 2 + coords[1] ** 2)
        b = tt.reshape(b, rgrid.shape)
        los = tt.reshape(coords[2], rgrid.shape)

        lc = self._compute_light_curve(
            b / orbit.r_star, rgrid / orbit.r_star, los / orbit.r_star
        )

        if texp is not None:
            stencil = tt.shape_padright(tt.shape_padleft(stencil, t.ndim), 1)
            lc = tt.sum(stencil * lc, axis=t.ndim)

        if use_in_transit:
            transit_model = tt.set_subtensor(transit_model[inds], lc)
            return transit_model
        else:
            return lc

    def _compute_light_curve(self, b, r, los=None):
        """Compute the light curve for a set of impact parameters and radii

        .. note:: The stellar radius is *not* included in this method so the
            coordinates should be in units of the star's radius.

        Args:
            b (tensor): A tensor of impact parameter values.
            r (tensor): A tensor of radius ratios with the same shape as ``b``.
            los (Optional[tensor]): The coordinates of the body along the
                line-of-sight. If ``los > 0`` the body is between the observer
                and the source.

        """
        b = as_tensor_variable(b)
        if los is None:
            los = tt.ones_like(b)
        lc = quad_limbdark_light_curve(self.c, b, r)
        return tt.switch(tt.gt(los, 0), lc, tt.zeros_like(lc))

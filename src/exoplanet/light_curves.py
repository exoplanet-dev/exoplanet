# -*- coding: utf-8 -*-

__all__ = [
    "LimbDarkLightCurve",
    "EclipsingBinaryLightCurve",
    "StarryLightCurve",
    "IntegratedLimbDarkLightCurve",
]

import warnings

import numpy as np
import theano
import theano.tensor as tt
from theano.gof import MissingInputError

from .citations import add_citations_to_model
from .theano_ops.starry.get_cl import GetClOp
from .theano_ops.starry.integrated_limbdark import IntegratedLimbDarkOp
from .theano_ops.starry.limbdark import LimbDarkOp
from .utils import eval_in_model

get_cl = GetClOp()
limbdark = LimbDarkOp()


class LimbDarkLightCurve:
    """A limb darkened light curve computed using starry

    Args:
        u (vector): A vector of limb darkening coefficients.

    """

    __citations__ = ("starry",)

    def __init__(self, u, model=None):
        add_citations_to_model(self.__citations__, model=model)
        self.u = tt.as_tensor_variable(u)
        u_ext = tt.concatenate([-1 + tt.zeros(1, dtype=self.u.dtype), self.u])
        self.c = get_cl(u_ext)
        self.c_norm = self.c / (np.pi * (self.c[0] + 2 * self.c[1] / 3))

    def get_light_curve(
        self,
        orbit=None,
        r=None,
        t=None,
        texp=None,
        oversample=7,
        order=0,
        use_in_transit=True,
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
                for each body in ``orbit``.
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

        r = tt.as_tensor_variable(r)
        r = tt.reshape(r, (r.size,))
        t = tt.as_tensor_variable(t)

        if use_in_transit:
            transit_model = tt.shape_padleft(
                tt.zeros_like(r), t.ndim
            ) + tt.shape_padright(tt.zeros_like(t), r.ndim)
            inds = orbit.in_transit(t, r=r, texp=texp)
            t = t[inds]

        if texp is None:
            tgrid = t
            rgrid = tt.shape_padleft(r, tgrid.ndim) + tt.shape_padright(
                tt.zeros_like(tgrid), r.ndim
            )
        else:
            texp = tt.as_tensor_variable(texp)

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

        coords = orbit.get_relative_position(tgrid)
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
                line-of-sight. If ``los < 0`` the body is between the observer
                and the source.

        """
        b = tt.as_tensor_variable(b)
        if los is None:
            los = tt.ones_like(b)
        return limbdark(self.c_norm, b, r, los)[0]


class StarryLightCurve(LimbDarkLightCurve):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "StarryLightCurve has been deprecated. "
            "Use LimbDarkLightCurve instead.",
            DeprecationWarning,
        )
        super(StarryLightCurve, self).__init__(*args, **kwargs)


class EclipsingBinaryLightCurve:
    def __init__(self, u_primary, u_secondary, flux_ratio, model=None):
        self.primary = LimbDarkLightCurve(u_primary, model=model)
        self.secondary = LimbDarkLightCurve(u_secondary, model=model)
        self.flux_ratio = tt.as_tensor_variable(flux_ratio)

    def get_light_curve(
        self,
        orbit=None,
        r=None,
        t=None,
        texp=None,
        oversample=7,
        order=0,
        use_in_transit=True,
    ):
        orbit2 = orbit._flip(r)
        lc1 = self.primary.get_light_curve(
            orbit=orbit,
            r=r,
            t=t,
            texp=texp,
            oversample=oversample,
            order=order,
            use_in_transit=use_in_transit,
        )
        lc2 = self.secondary.get_light_curve(
            orbit=orbit2,
            r=orbit.r_star,
            t=t,
            texp=texp,
            oversample=oversample,
            order=order,
            use_in_transit=use_in_transit,
        )
        return (lc1 + self.flux_ratio * lc2) / (1 + self.flux_ratio)


class IntegratedLimbDarkLightCurve:  # pragma: no cover

    """A limb darkened light curve computed using starry

    Args:
        u (vector): A vector of limb darkening coefficients.

    """

    __citations__ = ("starry",)

    def __init__(self, u, model=None):
        add_citations_to_model(self.__citations__, model=model)
        self.u = tt.as_tensor_variable(u)
        u_ext = tt.concatenate([-1 + tt.zeros(1, dtype=self.u.dtype), self.u])
        self.c = get_cl(u_ext)
        self.c_norm = self.c / (np.pi * (self.c[0] + 2 * self.c[1] / 3))

    @property
    def num_cl(self):
        # Try to compute the number of CLs
        try:
            func = theano.function([], self.c_norm.size)
            return int(func())
        except MissingInputError:
            pass
        try:
            return int(eval_in_model(self.c_norm.size))
        except (MissingInputError, TypeError):
            return -1

    def get_light_curve(
        self,
        orbit=None,
        r=None,
        t=None,
        texp=None,
        return_num_eval=False,
        **kwargs
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
                for each body in ``orbit``.
            t (tensor): The times where the light curve should be evaluated.
            texp (Optional[tensor]): The exposure time of each observation.
                This can be a scalar or a tensor with the same shape as ``t``.
                If ``texp`` is provided, ``t`` is assumed to indicate the
                timestamp at the *middle* of an exposure of length ``texp``.

        """
        if orbit is None:
            raise ValueError("missing required argument 'orbit'")
        if r is None:
            raise ValueError("missing required argument 'r'")
        if t is None:
            raise ValueError("missing required argument 't'")

        r = tt.as_tensor_variable(r)
        r = tt.reshape(r, (r.size,))
        t = tt.as_tensor_variable(t)

        def pad(arg):
            return arg
            # return tt.shape_padleft(arg, t.ndim) + tt.shape_padright(
            #     tt.zeros_like(t), arg.ndim
            # )

        rgrid = pad(r)
        if texp is None:
            coords = orbit.get_relative_position(t)
            b = tt.sqrt(coords[0] ** 2 + coords[1] ** 2)
            b = tt.reshape(b, rgrid.shape)
            los = tt.reshape(coords[2], rgrid.shape)
            return limbdark(
                self.c_norm, b / orbit.r_star, rgrid / orbit.r_star, los
            )[0]

        n = pad(orbit.n)
        sini = pad(orbit.sin_incl)
        cosi = pad(orbit.cos_incl)
        # texp = tt.as_tensor_variable(texp) + tt.zeros_like(rgrid)

        if orbit.ecc is None:
            aome2 = pad(-orbit.a)
            e = 0.0
            sinw = 0.0
            cosw = 0.0
            kwargs["circular"] = True
        else:
            aome2 = pad(-orbit.a * (1 - orbit.ecc ** 2))
            e = pad(orbit.ecc)
            sinw = pad(orbit.sin_omega)
            cosw = pad(orbit.cos_omega)
            kwargs["circular"] = False

        # Apply the time integrated op
        tgrid = tt.transpose(orbit._warp_times(t) - orbit.tref)
        texp = tt.as_tensor_variable(texp) + tt.zeros_like(tgrid)
        kwargs["Nc"] = kwargs.get("Nc", self.num_cl)
        op = IntegratedLimbDarkOp(**kwargs)
        res = op(
            self.c_norm,
            texp,
            tgrid,
            rgrid / orbit.r_star,
            n,
            aome2,
            sini,
            cosi,
            e,
            sinw,
            cosw,
        )
        if return_num_eval:
            return res[0], res[1]
        return res[0]

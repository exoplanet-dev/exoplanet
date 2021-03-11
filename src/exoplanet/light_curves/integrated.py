# -*- coding: utf-8 -*-

__all__ = ["IntegratedLimbDarkLightCurve"]

import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano
from aesara_theano_fallback.graph import fg

from ..citations import add_citations_to_model
from ..theano_ops.starry.integrated_limbdark import IntegratedLimbDarkOp
from ..utils import eval_in_model
from .limb_dark import get_cl, limbdark


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
        except fg.MissingInputError:
            pass
        try:
            return int(eval_in_model(self.c_norm.size))
        except (fg.MissingInputError, TypeError):
            return -1

    def get_light_curve(
        self,
        orbit=None,
        r=None,
        t=None,
        texp=None,
        return_num_eval=False,
        light_delay=False,
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
            coords = orbit.get_relative_position(t, light_delay=light_delay)
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

# -*- coding: utf-8 -*-

__all__ = ["ReboundOrbit"]

import theano.tensor as tt

from .constants import au_per_R_sun, day_per_yr_over_2pi
from .keplerian import KeplerianOrbit

try:
    from rebound_pymc3.integrate import IntegrateOp as ReboundOp
except ImportError:
    from ..theano_ops.rebound import ReboundOp


class ReboundOrbit(KeplerianOrbit):
    """An N-body system powered by the rebound integrator

    This takes all the same arguments as the :class:`KeplerianOrbit`, but
    these arguments define the orbital elements at some reference time (given
    by the ``rebound_t`` parameter). The positions and velocities of the bodies
    are then computed by numerically integrating the gravitational N-body
    system.

    ``rebound``-specific parameters can be provided as keyword arguments
    prefixed by ``rebound_``. These will then be applied to the
    ``rebound.Simulation`` object as properties. Therefore, if you want to
    change the integrator, you could use: ``rebound_integrator = "whfast"``,
    for example. All of these parameters are passed directly through to
    ``rebound`` except ``rebound_t`` (the reference time) which is converted
    from days to years over two pi (the default time units in ``rebound``).

    .. note:: ``exoplanet`` and ``rebound`` use different base units, but all
        of the unit conversions are handled behind the scenes in this object
        so that means that you should mostly use the ``exoplanet`` units when
        interacting with this class and you should be very cautious about
        setting the ``rebound_G`` argument. One example of a case where you'll
        need to use the ``rebound`` units is when you want to set the
        integrator step size using the ``rebound_dt`` parameter.

    """

    __citations__ = ("astropy", "rebound")

    def __init__(self, *args, **kwargs):
        rebound_args = dict(
            (k[8:], kwargs.pop(k))
            for k in list(kwargs.keys())
            if k.startswith("rebound_")
        )
        # rebound_args["G"] = rebound_G  # Consistent units
        rebound_args["t"] = rebound_args.get("t", 0.0) / day_per_yr_over_2pi
        self.rebound_initial_time = rebound_args["t"]
        self.rebound_op = ReboundOp(**rebound_args)
        super(ReboundOrbit, self).__init__(*args, **kwargs)

        if self.period.ndim:
            self.masses = tt.concatenate(
                [[self.m_star], self.m_planet + tt.zeros_like(self.period)]
            )
        else:
            self.masses = tt.stack([self.m_star, self.m_planet])

    def _get_position_and_velocity(self, t):
        pos, vel = super(ReboundOrbit, self)._get_position_and_velocity(
            self.rebound_initial_time
        )
        initial_coords = tt.concatenate(
            (pos * au_per_R_sun, vel * au_per_R_sun * day_per_yr_over_2pi),
            axis=-1,
        )
        coords, _ = self.rebound_op(
            self.masses,
            initial_coords,
            (t - self.rebound_initial_time) / day_per_yr_over_2pi,
        )

        # Deal with strange ordering and the need for things to be C contiguous
        coords = coords.dimshuffle(2, 0, 1)
        coords = coords + tt.zeros(coords.shape, dtype=coords.dtype)

        pos = coords[:3, :, :] / au_per_R_sun
        vel = coords[3:, :, :] / (day_per_yr_over_2pi * au_per_R_sun)
        return pos, vel

    def get_planet_position(self, t):
        """The planets' positions in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        pos, _ = self._get_position_and_velocity(t)
        if self.period.ndim:
            return pos[0, :, 1:], pos[1, :, 1:], pos[2, :, 1:]
        return pos[0, :, 1], pos[1, :, 1], pos[2, :, 1]

    def get_star_position(self, t):
        """The star's position in the barycentric frame

        .. note:: Unlike the :class:`KeplerianOrbit`, this will not return
            the contributions from each planet separately.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        pos, _ = self._get_position_and_velocity(t)
        return pos[0, :, 0], pos[1, :, 0], pos[2, :, 0]

    def get_relative_position(self, t):
        """The planets' positions relative to the star in the X,Y,Z frame.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        pos, _ = self._get_position_and_velocity(t)
        if self.period.ndim:
            pos = pos[:, :, 1:] - pos[:, :, 0][:, :, None]
        else:
            pos = pos[:, :, 1] - pos[:, :, 0]
        return pos[0], pos[1], pos[2]

    def get_planet_velocity(self, t):
        """Get the planets' velocity vectors

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.

        """
        _, vel = self._get_position_and_velocity(t)
        if self.period.ndim:
            return vel[0, :, 1:], vel[1, :, 1:], vel[2, :, 1:]
        return vel[0, :, 1], vel[1, :, 1], vel[2, :, 1]

    def get_star_velocity(self, t):
        """Get the star's velocity vector

        .. note:: Unlike the :class:`KeplerianOrbit`, this will not return
            the contributions from each planet separately.

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.

        """
        _, vel = self._get_position_and_velocity(t)
        return vel[0, :, 0], vel[1, :, 0], vel[2, :, 0]

    def get_relative_velocity(self, t):
        """The planets' velocity relative to the star

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``R_sun/day``.

        """
        _, vel = self._get_position_and_velocity(t)
        if self.period.ndim:
            vel = vel[:, :, 1:] - vel[:, :, 0][:, :, None]
        else:
            vel = vel[:, :, 1] - vel[:, :, 0]
        return vel[0], vel[1], vel[2]

    def _get_acceleration(self, *args, **kwargs):
        raise NotImplementedError(
            "the ReboundOrbit doesn't compute accelerations"
        )

    def get_radial_velocity(self, t, output_units=None):
        """Get the radial velocity of the star

        .. note:: The convention in exoplanet is that positive `z` points
            *towards* the observer. However, for consistency with radial
            velocity literature this method returns values where positive
            radial velocity corresponds to a redshift as expected.

        .. note:: Unlike the :class:`KeplerianOrbit` implementation, the
            semi-amplitude ``K`` cannot be used with the :class:`ReboundOrbit`.
            Also, the contributions of each planet are not returned separately;
            this will always return a single time series.

        Args:
            t: The times where the radial velocity should be evaluated.
            output_units (Optional): An AstroPy velocity unit. If not given,
                the output will be evaluated in ``m/s``. This is ignored if a
                value is given for ``K``.

        Returns:
            The reflex radial velocity evaluated at ``t`` in units of
            ``output_units``.

        """
        return super(ReboundOrbit, self).get_radial_velocity(
            t, K=None, output_units=output_units
        )

    def in_transit(self, t, r=0.0, texp=None):
        """This is a no-op and all points are assumed to be in transit"""
        return tt.arange(t.size)

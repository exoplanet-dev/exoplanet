# -*- coding: utf-8 -*-

__all__ = ["ReboundOrbit"]

import numpy as np
import astropy.units as u
import astropy.constants as c
import theano.tensor as tt

from .keplerian import KeplerianOrbit
from ..theano_ops.rebound import ReboundOp

day_per_yr_over_2pi = (
    (1.0 * u.au) ** (3 / 2)
    / (np.sqrt(c.G.to(u.au ** 3 / (u.M_sun * u.day ** 2)) * (1.0 * u.M_sun)))
).value
au_per_R_sun = u.R_sun.to(u.au)


class ReboundOrbit(KeplerianOrbit):

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
        pos = coords[:, :, :3] / au_per_R_sun
        vel = coords[:, :, 3:] / (day_per_yr_over_2pi * au_per_R_sun)
        return pos, vel

    def get_planet_position(self, t):
        pos, _ = self._get_position_and_velocity(t)
        if self.period.ndim:
            return pos[:, 1:, 0], pos[:, 1:, 1], pos[:, 1:, 2]
        return pos[:, 1, 0], pos[:, 1, 1], pos[:, 1, 2]

    def get_star_position(self, t):
        pos, _ = self._get_position_and_velocity(t)
        return pos[:, 0, 0], pos[:, 0, 1], pos[:, 0, 2]

    def get_relative_position(self, t):
        pos, _ = self._get_position_and_velocity(t)
        if self.period.ndim:
            pos = pos[:, 1:] - pos[:, 0][:, None, :]
            return pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]
        pos = pos[:, 1] - pos[:, 0]
        return pos[:, 0], pos[:, 1], pos[:, 2]

    def get_planet_velocity(self, t):
        _, vel = self._get_position_and_velocity(t)
        if self.period.ndim:
            return vel[:, 1:, 0], vel[:, 1:, 1], vel[:, 1:, 2]
        return vel[:, 1, 0], vel[:, 1, 1], vel[:, 1, 2]

    def get_star_velocity(self, t):
        _, vel = self._get_position_and_velocity(t)
        return vel[:, 0, 0], vel[:, 0, 1], vel[:, 0, 2]

    def get_relative_velocity(self, t):
        _, vel = self._get_position_and_velocity(t)
        if self.period.ndim:
            vel = vel[:, 1:] - vel[:, 0][:, None, :]
            return vel[:, :, 0], vel[:, :, 1], vel[:, :, 2]
        vel = vel[:, 1] - vel[:, 0]
        return vel[:, 0], vel[:, 1], vel[:, 2]

    def get_radial_velocity(self, t, K=None, output_units=None):
        if K is not None:
            raise ValueError(
                "semi amplitude K cannot be used with a ReboundOrbit"
            )
        return super(ReboundOrbit, self).get_radial_velocity(
            t, output_units=output_units
        )

    def in_transit(self, t, r=0.0, texp=None):
        return tt.arange(t.size)

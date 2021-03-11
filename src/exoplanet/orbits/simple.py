# -*- coding: utf-8 -*-

__all__ = ["SimpleTransitOrbit"]

import aesara_theano_fallback.tensor as tt
import numpy as np

from ..utils import as_tensor_variable


class SimpleTransitOrbit:
    """An orbit representing a set of planets transiting a common central

    This orbit is parameterized by the observables of a transiting system,
    period, phase, duration, and impact parameter.

    Args:
        period: The orbital period of the planets in days.
        t0: The midpoint time of a reference transit for each planet in days.
        b: The impact parameters of the orbits.
        duration: The durations of the transits in days.
        r_star: The radius of the star in ``R_sun``.

    """

    def __init__(self, period, duration, t0=0.0, b=0.0, r_star=1.0, ror=0):
        self.period = as_tensor_variable(period)
        self.t0 = as_tensor_variable(t0)
        self.b = as_tensor_variable(b)
        self.duration = as_tensor_variable(duration)
        self.r_star = as_tensor_variable(r_star)

        self._b_norm = self.b * self.r_star
        x2 = r_star ** 2 * ((1 + ror) ** 2 - b ** 2)
        self.speed = 2 * np.sqrt(x2) / duration

        self._half_period = 0.5 * self.period
        self._ref_time = self.t0 - self._half_period

    def get_star_position(self, t, light_delay=False):
        nothing = tt.zeros_like(as_tensor_variable(t))
        return nothing, nothing, nothing

    def get_planet_position(self, t, light_delay=False):
        return self.get_relative_position(t, light_delay=False)

    def get_relative_position(self, t, light_delay=False):
        """The planets' positions relative to the star

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        if light_delay:
            raise NotImplementedError(
                "Light travel time delay is not implemented for simple orbits"
            )
        dt = tt.mod(tt.shape_padright(t) - self._ref_time, self.period)
        dt -= self._half_period
        x = tt.squeeze(self.speed * dt)
        y = tt.squeeze(self._b_norm + tt.zeros_like(dt))
        m = tt.abs_(dt) < 0.5 * self.duration
        z = tt.squeeze(m * 1.0 - (~m) * 1.0)
        return x, y, z

    def get_planet_velocity(self, t):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def get_star_velocity(self, t):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def get_radial_velocity(self, t, output_units=None):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def in_transit(self, t, r=None, texp=None, light_delay=False):
        """Get a list of timestamps that are in transit

        Args:
            t (vector): A vector of timestamps to be evaluated.
            r (Optional): The radii of the planets.
            texp (Optional[float]): The exposure time.

        Returns:
            The indices of the timestamps that are in transit.

        """
        if light_delay:
            raise NotImplementedError(
                "Light travel time delay is not implemented for simple orbits"
            )
        dt = tt.mod(tt.shape_padright(t) - self._ref_time, self.period)
        dt -= self._half_period
        if r is None:
            tol = 0.5 * self.duration
        else:
            x = (r + self.r_star) ** 2 - self._b_norm ** 2
            tol = tt.sqrt(x) / self.speed
        if texp is not None:
            tol += 0.5 * texp
        mask = tt.any(tt.abs_(dt) < tol, axis=-1)
        return tt.arange(t.size)[mask]

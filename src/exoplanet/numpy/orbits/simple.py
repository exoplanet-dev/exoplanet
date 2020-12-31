# -*- coding: utf-8 -*-

__all__ = ["SimpleTransitOrbit"]

from .. import compat
from ..compat import numpy as np


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

    def __init__(
        self, *, period, duration, t0=0.0, b=0.0, r_star=1.0, r_planet=0.0
    ):
        self.period = compat.as_tensor(period)
        self.duration = compat.as_tensor(duration)
        self.t0 = compat.as_tensor(t0)
        self.b = compat.as_tensor(b)
        self.r_star = compat.as_tensor(r_star)
        self.r_planet = compat.as_tensor(r_planet)

        self._b_norm = self.b * self.r_star
        x2 = self.r_star ** 2 - self._b_norm ** 2
        self.speed = 2 * np.sqrt(x2) / self.duration
        self._half_period = 0.5 * self.period
        self._ref_time = self.t0 - self._half_period

    def get_star_position(self, t, light_delay=False):
        nothing = np.zeros_like(compat.as_tensor(t))
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
        dt = np.mod(
            compat.as_tensor(t)[..., None] - self._ref_time, self.period
        )
        dt -= self._half_period
        x = np.squeeze(self.speed * dt)
        y = np.squeeze(self._b_norm + np.zeros_like(dt))
        m = np.abs_(dt) < 0.5 * self.duration
        z = np.squeeze(m * 1.0 - (~m) * 1.0)
        return x, y, z

    def get_planet_velocity(self, t):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def get_star_velocity(self, t):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def get_radial_velocity(self, t, output_units=None):
        raise NotImplementedError("a SimpleTransitOrbit has no velocity")

    def in_transit(self, t, texp=None, light_delay=False):
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
        dt = np.mod(
            compat.as_tensor(t)[..., None] - self._ref_time, self.period
        )
        dt -= self._half_period
        x = (self.r_planet + self.r_star) ** 2 - self._b_norm ** 2
        tol = np.sqrt(x) / self.speed
        if texp is not None:
            tol += 0.5 * texp
        mask = np.any(compat.abs_(dt) < tol, axis=-1)
        return np.arange(t.size)[mask]

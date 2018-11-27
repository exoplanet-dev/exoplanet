# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerianOrbit", "get_true_anomaly"]

import numpy as np

import theano.tensor as tt

from astropy import constants
from astropy import units as u

from ..citations import add_citations_to_model
from ..theano_ops.kepler.solver import KeplerOp


class KeplerianOrbit(object):
    """A system of bodies on Keplerian orbits around a common central

    Given the input parameters, the values of all other parameters will be
    computed so a ``KeplerianOrbit`` instance will always have attributes for
    each argument. Note that the units of the computed attributes will all be
    in the standard units of this class (``R_sun``, ``M_sun``, and ``days``)
    except for ``rho_star`` which will be in ``g / cm^3``.

    There are only specific combinations of input parameters that can be used:

    1. First, either ``period`` or ``a`` must be given. If values are given
       for both parameters, then neither ``m_star`` or ``rho_star`` can be
       defined because the stellar density implied by each planet will be
       computed in ``rho_star``.
    2. Only one of ``incl`` and ``b`` can be given.
    3. If a value is given for ``ecc`` then ``omega`` must also be given.
    4. If no stellar parameters are given, the central body is assumed to be
       the sun. If only ``rho_star`` is defined, the radius of the central is
       assumed to be ``1 * R_sun``. Otherwise, at most two of ``m_star``,
       ``r_star``, and ``rho_star`` can be defined.

    Args:
        period: The orbital periods of the bodies in days.
        a: The semimajor axes of the orbits in ``R_sun``.
        t0: The time of a reference transit for each orbits in days.
        incl: The inclinations of the orbits in radians.
        b: The impact parameters of the orbits.
        ecc: The eccentricities of the orbits. Must be ``0 <= ecc < 1``.
        omega: The arguments of periastron for the orbits in radians.
        m_planet: The masses of the planets in units of ``m_planet_units``.
        m_star: The mass of the star in ``M_sun``.
        r_star: The radius of the star in ``R_sun``.
        rho_star: The density of the star in units of ``rho_star_units``.
        m_planet_units: An ``astropy.units`` compatible unit object giving the
            units of the planet masses. If not given, the default is ``M_sun``.
        rho_star_units: An ``astropy.units`` compatible unit object giving the
            units of the stellar density. If not given, the default is
            ``g / cm^3``.

    """

    __citations__ = ("astropy", )

    def __init__(self,
                 period=None, a=None, t0=0.0, incl=None, b=None,
                 ecc=None, omega=None, m_planet=0.0,
                 m_star=None, r_star=None, rho_star=None,
                 m_planet_units=None, rho_star_units=None,
                 model=None,
                 **kwargs):
        add_citations_to_model(self.__citations__, model=model)

        self.gcc_to_sun = (
            (constants.M_sun / constants.R_sun**3).to(u.g / u.cm**3).value)
        self.G_grav = constants.G.to(u.R_sun**3 / u.M_sun / u.day**2).value

        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.period = tt.as_tensor_variable(period)
        self.t0 = tt.as_tensor_variable(t0)
        self.m_planet = tt.as_tensor_variable(m_planet)
        if m_planet_units is not None:
            self.m_planet *= (1 * m_planet_units).to(u.M_sun).value

        self.a, self.period, self.rho_star, self.r_star, self.m_star = \
            self._get_consistent_inputs(a, period, rho_star, r_star, m_star,
                                        rho_star_units)
        self.m_total = self.m_star + self.m_planet

        self.n = 2 * np.pi / self.period
        self.a_star = self.a * self.m_planet / self.m_total
        self.a_planet = -self.a * self.m_star / self.m_total

        if incl is None:
            if b is None:
                self.incl = tt.as_tensor_variable(0.5 * np.pi)
                self.b = tt.as_tensor_variable(0.0)
            else:
                self.b = tt.as_tensor_variable(b)
                self.incl = tt.arccos(self.b / self.a_planet)
        else:
            if b is not None:
                raise ValueError("only one of 'incl' and 'b' can be given")
            self.incl = tt.as_tensor_variable(incl)
            self.b = self.a_planet * tt.cos(self.incl)

        self.K0 = self.n * self.a / self.m_total
        self.cos_incl = tt.cos(self.incl)
        self.sin_incl = tt.sin(self.incl)

        # Eccentricity
        if ecc is None:
            self.ecc = None
            self.tref = self.t0 - 0.5 * np.pi / self.n
        else:
            self.ecc = tt.as_tensor_variable(ecc)
            if omega is None:
                raise ValueError("both e and omega must be provided")
            self.omega = tt.as_tensor_variable(omega)

            self.cos_omega = tt.cos(self.omega)
            self.sin_omega = tt.sin(self.omega)

            E0 = 2.0 * tt.arctan2(tt.sqrt(1.0-self.ecc)*self.cos_omega,
                                  tt.sqrt(1.0+self.ecc)*(1.0+self.sin_omega))
            self.tref = self.t0 - (E0 - self.ecc * tt.sin(E0)) / self.n

            self.K0 /= tt.sqrt(1 - self.ecc**2)

    def _get_consistent_inputs(self, a, period, rho_star, r_star, m_star,
                               rho_star_units):
        if a is None and period is None:
            raise ValueError("values must be provided for at least one of a "
                             "and period")

        if a is not None:
            a = tt.as_tensor_variable(a)
        if period is not None:
            period = tt.as_tensor_variable(period)

        # Compute the implied density if a and period are given
        if a is not None and period is not None:
            if rho_star is not None or m_star is not None:
                raise ValueError("if both a and period are given, you can't "
                                 "also define rho_star or m_star")
            if r_star is None:
                r_star = 1.0
            rho_star = 3*np.pi*(a / r_star)**3 / (self.G_grav*period**2)
            rho_star -= 3*self.m_planet/(4*np.pi*r_star**3)
            rho_star_units = None

        # Make sure that the right combination of stellar parameters are given
        if r_star is None and m_star is None:
            r_star = 1.0
            if rho_star is None:
                m_star = 1.0
        if sum(arg is None for arg in (rho_star, r_star, m_star)) != 1:
            raise ValueError("values must be provided for exactly two of "
                             "rho_star, m_star, and r_star")

        if rho_star is not None:
            rho_star = tt.as_tensor_variable(rho_star)
            if rho_star_units is not None:
                rho_star *= (1 * rho_star_units).to(u.M_sun / u.R_sun**3).value
            else:
                rho_star /= self.gcc_to_sun
        if r_star is not None:
            r_star = tt.as_tensor_variable(r_star)
        if m_star is not None:
            m_star = tt.as_tensor_variable(m_star)

        # Work out the stellar parameters
        if rho_star is None:
            rho_star = 3*m_star/(4*np.pi*r_star**3)
        elif r_star is None:
            r_star = (3*m_star/(4*np.pi*rho_star))**(1/3)
        else:
            m_star = 4*np.pi*r_star**3*rho_star/3

        # Work out the planet parameters
        if a is None:
            a = (self.G_grav*(m_star+self.m_planet)*period**2 /
                 (4*np.pi**2))**(1./3)
        elif period is None:
            period = 2*np.pi*a**(3/2)/(
                tt.sqrt(self.G_grav*(m_star+self.m_planet)))

        return a, period, rho_star * self.gcc_to_sun, r_star, m_star

    def _rotate_vector(self, x, y):
        if self.ecc is None:
            a = x
            b = y
        else:
            a = self.cos_omega * x - self.sin_omega * y
            b = self.sin_omega * x + self.cos_omega * y
        return (a, self.cos_incl * b, self.sin_incl * b)

    def _warp_times(self, t):
        return tt.shape_padright(t)

    def _get_true_anomaly(self, t):
        M = (self._warp_times(t) - self.tref) * self.n
        if self.ecc is None:
            return M
        _, f = self.kepler_op(M, self.ecc + tt.zeros_like(M))
        return f

    def _get_position(self, a, t):
        f = self._get_true_anomaly(t)
        cosf = tt.cos(f)
        if self.ecc is None:
            r = a
        else:
            r = a * (1.0 - self.ecc**2) / (1 + self.ecc * cosf)
        return self._rotate_vector(r * cosf, r * tt.sin(f))

    def get_planet_position(self, t):
        """The planets' positions in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            x, y, z: The components of the position vector at ``t`` in units
                of ``R_sun``.

        """
        return tuple(tt.squeeze(x)
                     for x in self._get_position(self.a_planet, t))

    def get_star_position(self, t):
        """The star's position in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            x, y, z: The components of the position vector at ``t`` in units
                of ``R_sun``.

        """
        return tuple(tt.squeeze(x)
                     for x in self._get_position(self.a_star, t))

    def get_relative_position(self, t):
        """The planets' positions relative to the star

        Args:
            t: The times where the position should be evaluated.

        Returns:
            x, y, z: The components of the position vector at ``t`` in units
                of ``R_sun``.

        """
        star = self._get_position(self.a_star, t)
        planet = self._get_position(self.a_planet, t)
        return tuple(tt.squeeze(b-tt.shape_padright(tt.sum(a, axis=-1)))
                     for a, b in zip(star, planet))

    def _get_velocity(self, m, t):
        f = self._get_true_anomaly(t)
        K = self.K0 * m
        if self.ecc is None:
            return self._rotate_vector(-K*tt.sin(f), K*tt.cos(f))
        return self._rotate_vector(-K*tt.sin(f), K*(tt.cos(f) + self.ecc))

    def get_planet_velocity(self, t):
        """Get the planets' velocity vector

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            vx, vy, vz: The components of the velocity vector at ``t`` in units
                of ``M_sun/day``.

        """
        return tuple(tt.squeeze(x)
                     for x in self._get_velocity(-self.m_star, t))

    def get_star_velocity(self, t):
        """Get the star's velocity vector

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            vx, vy, vz: The components of the velocity vector at ``t`` in units
                of ``M_sun/day``.

        """
        return tuple(tt.squeeze(x)
                     for x in self._get_velocity(self.m_planet, t))

    def get_radial_velocity(self, t, output_units=None):
        """Get the radial velocity of the star

        Args:
            t: The times where the radial velocity should be evaluated.
            output_units (Optional): An AstroPy velocity unit. If not given,
                the output will be evaluated in ``m/s``.

        Returns:
            vrad: The radial velocity evaluated at ``t`` in units of
                ``output_units``.

        """
        if output_units is None:
            output_units = u.m / u.s
        conv = (1 * u.R_sun / u.day).to(output_units).value
        v = self.get_star_velocity(t)
        return conv * v[2]

    def approx_in_transit(self, t, r=0.0, texp=None, duration_factor=3):
        """Get a list of timestamps that are expected to be in transit

        Args:
            t (vector): A vector of timestamps to be evaluated.
            r (Optional): The radii of the planets.
            texp (Optional[float]): The exposure time.
            duration_factor (Optional[float]): The factor by which to multiply
                the approximate duration when computing the in transit points.
                Larger values will be more conservative and might be needed for
                large planets or very eccentric orbits.

        Returns:
            inds (vector): The indices of the timestamps that are expected to
                be in transit.

        """
        # Estimate the maximum duration of the transit using the equations
        # from Winn (2010)
        arg = (self.r_star + r) / (-self.a_planet)
        max_dur = self.period * tt.arcsin(arg) / np.pi
        if self.ecc is not None:
            max_dur *= tt.sqrt(1-self.ecc**2) / (1+self.ecc*self.sin_omega)

        # Wrap the times into time since transit
        hp = 0.5 * self.period
        dt = tt.mod(self._warp_times(t) - self.t0 + hp, self.period) - hp

        # Estimate the data points that are within the maximum duration of the
        # transit
        tol = 0.5 * duration_factor * max_dur
        if texp is not None:
            tol += 0.5 * texp
        mask = tt.any(tt.abs_(dt) < tol, axis=-1)

        return tt.arange(t.size)[mask]


def get_true_anomaly(M, e, **kwargs):
    """Get the true anomaly for a tensor of mean anomalies and eccentricities

    Args:
        M: The mean anomaly.
        e: The eccentricity. This should have the same shape as ``M``.

    Returns:
        The true anomaly of the orbit.

    """
    return KeplerOp()(M, e)[1]

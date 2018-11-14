# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerianOrbit"]

import numpy as np

import theano.tensor as tt

from astropy import constants
from astropy import units as u

from ..theano_ops.kepler.solver import KeplerOp

gcc_to_sun = (constants.M_sun / constants.R_sun**3).to(u.g / u.cm**3).value
G_grav = constants.G.to(u.R_sun**3 / u.M_sun / u.day**2).value


class KeplerianOrbit(object):

    def __init__(self,
                 period=None, a=None, rho_star=None,
                 t0=0.0, incl=0.5*np.pi,
                 m_star=None, r_star=None,
                 ecc=None, omega=None,
                 m_planet=0.0, **kwargs):
        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.period = tt.as_tensor_variable(period)
        self.t0 = tt.as_tensor_variable(t0)
        self.incl = tt.as_tensor_variable(incl)
        self.m_planet = tt.as_tensor_variable(m_planet)

        self.a, self.period, self.rho_star, self.r_star, self.m_star = \
            self._get_consistent_inputs(a, period, rho_star, r_star, m_star)
        self.m_total = self.m_star + self.m_planet

        self.n = 2 * np.pi / self.period
        self.a_star = self.a * self.m_planet / self.m_total
        self.a_planet = -self.a * self.m_star / self.m_total

        self.K0 = self.n * self.a / self.m_total
        self.cos_incl = tt.cos(incl)
        self.sin_incl = tt.sin(incl)

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

    def _get_consistent_inputs(self, a, period, rho_star, r_star, m_star):
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
            rho_star = 3*np.pi*(a / r_star)**3 / (G_grav*period**2)
            rho_star -= 3*self.m_planet/(4*np.pi*r_star**3)

        # Make sure that the right combination of stellar parameters are given
        if r_star is None and m_star is None:
            r_star = 1.0
            if rho_star is None:
                m_star = 1.0
        if sum(arg is None for arg in (rho_star, r_star, m_star)) != 1:
            raise ValueError("values must be provided for exactly two of "
                             "rho_star, m_star, and r_star")

        if rho_star is not None:
            # Convert density to M_sun / R_sun^3
            rho_star = tt.as_tensor_variable(rho_star) / gcc_to_sun
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
            a = (G_grav*(m_star+self.m_planet)*period**2/(4*np.pi**2))**(1./3)
        elif period is None:
            period = 2*np.pi*a**(3/2)/(tt.sqrt(G_grav*(m_star+self.m_planet)))

        return a, period, rho_star * gcc_to_sun, r_star, m_star

    def _rotate_vector(self, x, y):
        if self.ecc is None:
            a = x
            b = y
        else:
            a = self.cos_omega * x - self.sin_omega * y
            b = self.sin_omega * x + self.cos_omega * y
        return (a, self.cos_incl * b, self.sin_incl * b)

    def _get_true_anomaly(self, t):
        M = (tt.shape_padright(t) - self.tref) * self.n
        if self.ecc is None:
            return tt.squeeze(M)
        E = self.kepler_op(M, self.ecc + tt.zeros_like(M))
        f = 2.0 * tt.arctan2(tt.sqrt(1.0 + self.ecc) * tt.tan(0.5*E),
                             tt.sqrt(1.0 - self.ecc) + tt.zeros_like(E))
        return tt.squeeze(f)

    def _get_position(self, a, t):
        f = self._get_true_anomaly(t)
        cosf = tt.cos(f)
        if self.ecc is None:
            r = a
        else:
            r = a * (1.0 - self.ecc**2) / (1 + self.ecc * cosf)
        return self._rotate_vector(r * cosf, r * tt.sin(f))

    def get_planet_position(self, t):
        """The planets' positions in Solar radii"""
        return self._get_position(self.a_planet, t)

    def get_star_position(self, t):
        """The star's position in Solar radii"""
        return self._get_position(self.a_star, t)

    def get_relative_position(self, t):
        """The planets' positions relative to the star"""
        star = self._get_position(self.a_star, t)
        planet = self._get_position(self.a_planet, t)
        return tuple(b-a for a, b in zip(star, planet))

    def _get_velocity(self, m, t):
        f = self._get_true_anomaly(t)
        K = self.K0 * m
        return self._rotate_vector(-K*tt.sin(f), K*(tt.cos(f) + self.ecc))

    def get_planet_velocity(self, t):
        """The planets' velocities in R_sun / day"""
        return self._get_velocity(-self.m_star, t)

    def get_star_velocity(self, t):
        """The star's velocity in R_sun / day"""
        return self._get_velocity(self.m_planet, t)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["estimate_minimum_mass"]

import numpy as np

import astropy.units as u


def _get_design_matrix(periods, t0s, x):
    if t0s is not None:
        return np.vstack([
            np.cos(2*np.pi*(x - (t0s[i] - 0.25*periods[i])) / periods[i])
            for i in range(len(periods))
        ] + [np.ones(len(x))]).T
    return np.concatenate([
        (np.sin(2*np.pi*x / periods[i]),
         np.cos(2*np.pi*x / periods[i]))
        for i in range(len(periods))
    ] + [np.ones((1, len(x)))], axis=0).T


def estimate_minimum_mass(periods, x, y, yerr=None, t0s=None, m_star=1):
    """Estimate the minimum mass(es) for planets in an RV series

    Args:
        periods: The periods of the planets. Assumed to be in ``days`` if not
            an AstroPy Quantity.
        x: The observation times. Assumed to be in ``days`` if not an AstroPy
            Quantity.
        y: The radial velocities. Assumed to be in ``m/s`` if not an AstroPy
            Quantity.
        yerr (Optional): The uncertainty on ``y``.
        t0s (Optional): The time of a reference transit for each planet, if
            known.
        m_star (Optional): The mass of the star. Assumed to be in ``M_sun``
            if not an AstroPy Quantity.

    Returns:
        msini: An estimate of the minimum mass of each planet as an AstroPy
            Quantity with units of ``M_jupiter``.

    """
    if yerr is None:
        ivar = np.ones_like(y)
    else:
        ivar = 1.0 / yerr**2

    m_star = u.Quantity(m_star, unit=u.M_sun)
    periods = u.Quantity(np.atleast_1d(periods), unit=u.day)
    if t0s is not None:
        t0s = u.Quantity(np.atleast_1d(t0s), unit=u.day).value
    x = u.Quantity(np.atleast_1d(x), unit=u.day)
    y = u.Quantity(np.atleast_1d(y), unit=u.m / u.s)
    ivar = u.Quantity(np.atleast_1d(ivar), unit=(u.s / u.m) ** 2)

    D = _get_design_matrix(periods.value, t0s, x.value)
    w = np.linalg.solve(np.dot(D.T, D*ivar.value[:, None]),
                        np.dot(D.T, y.value*ivar.value))
    if t0s is not None:
        K = w[:-1]
    else:
        w = w[:-1]
        K = np.sqrt(w[::2]**2 + w[1::2]**2)

    m_J = K / 28.4329 * m_star.value**(2./3)
    m_J *= (periods.to(u.year)).value**(1./3)

    return m_J * u.M_jupiter

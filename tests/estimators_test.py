# -*- coding: utf-8 -*-

import astropy.units as u
import numpy as np

from exoplanet.estimators import (
    autocorr_estimator,
    bls_estimator,
    estimate_minimum_mass,
    estimate_semi_amplitude,
    lomb_scargle_estimator,
)
from exoplanet.orbits import KeplerianOrbit


def test_estimate_semi_amplitude(seed=9502):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 10, 500))
    period = 2.345
    amp = 4.5
    y = amp * np.sin(2 * np.pi * t / period)
    est = estimate_semi_amplitude(period, t, y)
    assert np.allclose(est, amp)

    est = estimate_semi_amplitude(
        period, t, y, yerr=np.ones_like(t), t0s=0.5 * period
    )
    assert np.allclose(est, amp)


def test_estimate_minimum_mass(seed=9502):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 10, 500))
    period = 2.345
    t0 = 0.5
    orbit = KeplerianOrbit(period=period, t0=t0, m_planet=0.01, incl=0.8)
    y = orbit.get_radial_velocity(t).eval()
    m1 = (orbit.m_planet * orbit.sin_incl).eval()
    m2 = estimate_minimum_mass(period, t, y).to(u.M_sun).value
    m3 = estimate_minimum_mass(period, t, y, t0s=t0).to(u.M_sun).value
    assert np.abs((m1 - m2) / m1) < 0.01
    assert np.abs((m1 - m3) / m1) < 0.01


def test_lomb_scargle_estimator(seed=9502):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 10, 500))
    period = 2.345
    amp = 4.5
    y = amp * np.sin(2 * np.pi * t / period)
    results = lomb_scargle_estimator(
        t, y, min_period=1, max_period=10, filter_period=10.0
    )
    assert np.abs(period - results["peaks"][0]["period"]) / period < 0.001


def test_autocorr_estimator(seed=9502):
    np.random.seed(seed)
    t = np.linspace(0, 10, 500)
    period = 2.345
    amp = 4.5
    y = amp * np.sin(2 * np.pi * t / period)
    results = autocorr_estimator(
        t, y, min_period=0.01, max_period=10, smooth=0.0
    )
    assert np.abs(period - results["peaks"][0]["period"]) / period < 0.01


def test_bls_estimator(seed=721):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 27, 5000))
    y = 1e-3 * np.random.randn(len(t))
    period = 2.345
    t0 = 1.0
    depth = 0.1
    duration = 0.2
    y[
        np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
        < 0.5 * duration
    ] -= depth
    results = bls_estimator(t, y)
    assert np.abs(period - results["peaks"][0]["period"]) / period < 0.01

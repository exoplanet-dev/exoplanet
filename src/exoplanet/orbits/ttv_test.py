# -*- coding: utf-8 -*-

import numpy as np
import theano

from .keplerian import KeplerianOrbit
from .ttv import TTVOrbit, compute_expected_transit_times


def test_compute_expected_transit_times():
    periods = [10.5, 56.34]
    t0s = [45.3, 48.1]
    min_time = 456.023
    max_time = 595.23
    transit_times = compute_expected_transit_times(
        min_time, max_time, periods, t0s
    )
    for period, t0, times in zip(periods, t0s, transit_times):
        assert np.all(min_time <= times)
        assert np.all(times <= max_time)
        assert times[0] - period < min_time
        assert times[-1] + period > max_time


def test_consistency(seed=6934104):
    np.random.seed(seed)
    periods = [10.5, 56.34]
    t0s = [45.3, 48.1]
    min_time = 456.023
    max_time = 595.23
    expected_times = compute_expected_transit_times(
        min_time, max_time, periods, t0s
    )
    ttvs = [0.01 * np.random.randn(len(t)) for t in expected_times]
    orbit = TTVOrbit(
        period=periods, t0=[t[0] for t in expected_times], ttvs=ttvs
    )
    calc_times = theano.function([], orbit.transit_times)()
    for i in range(len(expected_times)):
        assert np.allclose(calc_times[i], expected_times[i] + ttvs[i])

    orbit1 = TTVOrbit(transit_times=orbit.transit_times)
    orbit2 = TTVOrbit(period=orbit1.period, t0=orbit1.t0, ttvs=orbit1.ttvs)
    for i in range(len(expected_times)):
        assert np.allclose(
            orbit1.transit_times[i].eval(), orbit2.transit_times[i].eval()
        )
        assert np.allclose(orbit1.ttvs[i].eval(), orbit2.ttvs[i].eval())


def test_no_ttvs():
    periods = [10.5, 56.34]
    t0s = [45.3, 48.1]
    min_time = 456.023
    max_time = 595.23
    time = np.linspace(min_time, max_time, 5000)
    expected_times = compute_expected_transit_times(
        min_time, max_time, periods, t0s
    )

    orbit0 = KeplerianOrbit(period=periods, t0=t0s)

    orbit1 = TTVOrbit(
        period=periods,
        t0=[t[0] for t in expected_times],
        ttvs=[np.zeros_like(t) for t in expected_times],
    )

    orbit2 = TTVOrbit(transit_times=expected_times)

    for arg in [
        "get_relative_position",
        "get_star_position",
        "get_planet_position",
        "get_relative_velocity",
        "get_star_velocity",
        "get_planet_velocity",
        "get_radial_velocity",
    ]:
        expect = theano.function([], getattr(orbit0, arg)(time))()
        calc1 = theano.function([], getattr(orbit1, arg)(time))()
        calc2 = theano.function([], getattr(orbit2, arg)(time))()

        assert np.allclose(expect, calc1), arg
        assert np.allclose(expect, calc2), arg

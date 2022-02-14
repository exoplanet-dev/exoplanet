# -*- coding: utf-8 -*-

import aesara_theano_fallback.tensor as tt
import astropy.units as u
import numpy as np
import pytest
from aesara_theano_fallback import aesara as theano
from astropy.constants import c
from scipy.optimize import minimize

from exoplanet.orbits.keplerian import (
    KeplerianOrbit,
    _get_consistent_inputs,
    get_aor_from_transit_duration,
)
from exoplanet.units import with_unit


def test_sky_coords():
    _rsky = pytest.importorskip("batman._rsky")

    t = np.linspace(-100, 100, 1000)

    t0, period, a, e, omega, incl = (
        x.flatten()
        for x in np.meshgrid(
            np.linspace(-5.0, 5.0, 2),
            np.exp(np.linspace(np.log(5.0), np.log(50.0), 3)),
            np.linspace(50.0, 100.0, 2),
            np.linspace(0.0, 0.9, 5),
            np.linspace(-np.pi, np.pi, 3),
            np.arccos(np.linspace(0, 1, 5)[:-1]),
        )
    )
    r_batman = np.empty((len(t), len(t0)))

    for i in range(len(t0)):
        r_batman[:, i] = _rsky._rsky(
            t, t0[i], period[i], a[i], incl[i], e[i], omega[i], 1, 1
        )
    m = r_batman < 100.0
    assert m.sum() > 0

    orbit = KeplerianOrbit(
        period=period, a=a, t0=t0, ecc=e, omega=omega, incl=incl
    )
    func = theano.function([], orbit.get_relative_position(t))
    x, y, z = func()
    r = np.sqrt(x**2 + y**2)

    # Make sure that the in-transit impact parameter matches batman
    assert np.allclose(r_batman[m], r[m], atol=2e-5)

    # In-transit should correspond to positive z in our parameterization
    assert np.all(z[m] > 0)

    # Therefore, when batman doesn't see a transit we shouldn't be transiting
    no_transit = z[~m] < 0
    no_transit |= r[~m] > 2
    assert np.all(no_transit)


def test_center_of_mass():
    t = np.linspace(0, 100, 1000)
    m_planet = np.array([0.5, 0.1])
    m_star = 1.45
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.0,
        t0=np.array([0.5, 17.4]),
        period=np.array([100.0, 37.3]),
        ecc=np.array([0.1, 0.8]),
        omega=np.array([0.5, 1.3]),
        Omega=np.array([0.0, 1.0]),
        incl=np.array([0.25 * np.pi, 0.3 * np.pi]),
        m_planet=m_planet,
    )

    planet_coords = theano.function([], orbit.get_planet_position(t))()
    star_coords = theano.function([], orbit.get_star_position(t))()

    com = np.sum(
        (
            m_planet[None, :] * np.array(planet_coords)
            + m_star * np.array(star_coords)
        )
        / (m_star + m_planet)[None, :],
        axis=0,
    )
    assert np.allclose(com, 0.0)


def test_velocity():
    t_tensor = tt.dvector()
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.0,
        t0=0.5,
        period=100.0,
        ecc=0.1,
        omega=0.5,
        Omega=1.0,
        incl=0.25 * np.pi,
        m_planet=m_planet,
    )

    star_pos = orbit.get_star_position(t_tensor)
    star_vel = theano.function([], orbit.get_star_velocity(t))()
    star_vel_expect = np.empty_like(star_vel)
    for i in range(3):
        g = theano.grad(tt.sum(star_pos[i]), t_tensor)
        star_vel_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(star_vel, star_vel_expect)

    planet_pos = orbit.get_planet_position(t_tensor)
    planet_vel = theano.function([], orbit.get_planet_velocity(t))()
    planet_vel_expect = np.empty_like(planet_vel)
    for i in range(3):
        g = theano.grad(tt.sum(planet_pos[i]), t_tensor)
        planet_vel_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(planet_vel, planet_vel_expect)

    pos = orbit.get_relative_position(t_tensor)
    vel = np.array(theano.function([], orbit.get_relative_velocity(t))())
    vel_expect = np.empty_like(vel)
    for i in range(3):
        g = theano.grad(tt.sum(pos[i]), t_tensor)
        vel_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(vel, vel_expect)


def test_radial_velocity():
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.0,
        t0=0.5,
        period=100.0,
        ecc=0.1,
        omega=0.5,
        Omega=1.0,
        incl=0.25 * np.pi,
        m_planet=m_planet,
    )

    rv1 = orbit.get_radial_velocity(t, output_units=u.R_sun / u.day).eval()
    rv2 = orbit.get_radial_velocity(
        t, K=orbit.K0 * orbit.m_planet * orbit.sin_incl
    ).eval()
    assert np.allclose(rv1, rv2)


def test_acceleration():
    t_tensor = tt.dvector()
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.0,
        t0=0.5,
        period=100.0,
        ecc=0.1,
        omega=0.5,
        incl=0.25 * np.pi,
        m_planet=m_planet,
    )

    star_vel = orbit.get_star_velocity(t_tensor)
    star_acc = theano.function([], orbit.get_star_acceleration(t))()
    star_acc_expect = np.empty_like(star_acc)
    for i in range(3):
        g = theano.grad(tt.sum(star_vel[i]), t_tensor)
        star_acc_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(star_acc, star_acc_expect)

    planet_vel = orbit.get_planet_velocity(t_tensor)
    planet_acc = theano.function([], orbit.get_planet_acceleration(t))()
    planet_acc_expect = np.empty_like(planet_acc)
    for i in range(3):
        g = theano.grad(tt.sum(planet_vel[i]), t_tensor)
        planet_acc_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(planet_acc, planet_acc_expect)

    vel = orbit.get_relative_velocity(t_tensor)
    acc = theano.function([], orbit.get_relative_acceleration(t))()
    acc_expect = np.empty_like(acc)
    for i in range(3):
        g = theano.grad(tt.sum(vel[i]), t_tensor)
        acc_expect[i] = theano.function([t_tensor], g)(t)
    assert np.allclose(acc, acc_expect)


def test_flip():
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit1 = KeplerianOrbit(
        m_star=m_star,
        r_star=1.1,
        t0=0.5,
        period=100.0,
        ecc=0.3,
        omega=0.5,
        Omega=1.0,
        incl=0.25 * np.pi,
        m_planet=m_planet,
    )
    orbit2 = orbit1._flip(0.7)

    x1, y1, z1 = theano.function([], orbit1.get_star_position(t))()
    x2, y2, z2 = theano.function([], orbit2.get_planet_position(t))()
    assert np.allclose(x1, x2, atol=1e-5)
    assert np.allclose(y1, y2, atol=1e-5)
    assert np.allclose(z1, z2, atol=1e-5)

    x1, y1, z1 = theano.function([], orbit1.get_planet_position(t))()
    x2, y2, z2 = theano.function([], orbit2.get_star_position(t))()
    assert np.allclose(x1, x2, atol=1e-5)
    assert np.allclose(y1, y2, atol=1e-5)
    assert np.allclose(z1, z2, atol=1e-5)


def test_flip_circular():
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit1 = KeplerianOrbit(
        m_star=m_star,
        r_star=1.1,
        t0=0.5,
        period=100.0,
        Omega=1.0,
        incl=0.25 * np.pi,
        m_planet=m_planet,
    )
    orbit2 = orbit1._flip(0.7)

    x1, y1, z1 = theano.function([], orbit1.get_star_position(t))()
    x2, y2, z2 = theano.function([], orbit2.get_planet_position(t))()
    assert np.allclose(x1, x2, atol=1e-5)
    assert np.allclose(y1, y2, atol=1e-5)
    assert np.allclose(z1, z2, atol=1e-5)

    x1, y1, z1 = theano.function([], orbit1.get_planet_position(t))()
    x2, y2, z2 = theano.function([], orbit2.get_star_position(t))()
    assert np.allclose(x1, x2, atol=1e-5)
    assert np.allclose(y1, y2, atol=1e-5)
    assert np.allclose(z1, z2, atol=1e-5)


def test_in_transit():
    t = np.linspace(-20, 20, 1000)
    m_planet = np.array([0.3, 0.5])
    m_star = 1.45
    r_star = 1.5
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=r_star,
        t0=np.array([0.5, 17.4]),
        period=np.array([10.0, 5.3]),
        ecc=np.array([0.1, 0.8]),
        omega=np.array([0.5, 1.3]),
        Omega=np.array([0.0, 1.0]),
        m_planet=m_planet,
    )

    r_pl = np.array([0.1, 0.03])
    coords = theano.function([], orbit.get_relative_position(t))()
    r2 = coords[0] ** 2 + coords[1] ** 2
    inds = theano.function([], orbit.in_transit(t, r=r_pl))()

    m = np.isin(np.arange(len(t)), inds)
    in_ = r2[inds] <= ((r_star + r_pl) ** 2)[None, :]
    in_ &= coords[2][inds] > 0
    assert np.all(np.any(in_, axis=1))

    out = r2[~m] > ((r_star + r_pl) ** 2)[None, :]
    out |= coords[2][~m] <= 0
    assert np.all(out)


def test_in_transit_circ():
    t = np.linspace(-20, 20, 1000)
    m_planet = np.array([0.3, 0.5])
    m_star = 1.45
    r_star = 1.5
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=r_star,
        t0=np.array([0.5, 17.4]),
        period=np.array([10.0, 5.3]),
        ecc=np.array([0.0, 0.0]),
        omega=np.array([0.0, 0.0]),
        m_planet=m_planet,
    )
    orbit_circ = KeplerianOrbit(
        m_star=m_star,
        r_star=r_star,
        t0=np.array([0.5, 17.4]),
        period=np.array([10.0, 5.3]),
        m_planet=m_planet,
    )

    r_pl = np.array([0.1, 0.03])
    inds = theano.function([], orbit.in_transit(t, r=r_pl))()
    inds_circ = theano.function([], orbit_circ.in_transit(t, r=r_pl))()
    assert np.all(inds == inds_circ)


def test_small_star():
    _rsky = pytest.importorskip("batman._rsky")

    m_star = 0.151
    r_star = 0.189
    period = 0.4626413
    t0 = 0.2
    b = 0.5
    ecc = 0.1
    omega = 0.1
    t = np.linspace(0, period, 500)

    orbit = KeplerianOrbit(
        r_star=r_star,
        m_star=m_star,
        period=period,
        t0=t0,
        b=b,
        ecc=ecc,
        omega=omega,
    )
    a = orbit.a.eval()
    incl = orbit.incl.eval()

    r_batman = _rsky._rsky(t, t0, period, a, incl, ecc, omega, 1, 1)
    m = r_batman < 100.0
    assert m.sum() > 0

    func = theano.function([], orbit.get_relative_position(t))
    x, y, z = func()
    r = np.sqrt(x**2 + y**2)

    # Make sure that the in-transit impact parameter matches batman
    assert np.allclose(r_batman[m], r[m], atol=2e-5)


def test_impact():
    m_star = 0.151
    r_star = 0.189
    period = 0.4626413
    t0 = 0.2
    b = 0.5
    ecc = 0.8
    omega = 0.1

    orbit = KeplerianOrbit(
        r_star=r_star,
        m_star=m_star,
        period=period,
        t0=t0,
        b=b,
        ecc=ecc,
        omega=omega,
    )
    coords = orbit.get_relative_position(t0)
    assert np.allclose(
        (tt.sqrt(coords[0] ** 2 + coords[1] ** 2) / r_star).eval(), b
    )
    assert coords[2].eval() > 0


def test_consistent_coords():
    import astropy.constants as c
    import astropy.units as u

    au_to_R_sun = (c.au / c.R_sun).value
    a_ang = 0.324  # arcsec
    parallax = 24.05  # milliarcsec
    dpc = 1e3 / parallax
    a = a_ang * dpc  # au
    P = 28.8 * 365.25  # days

    # kappa = a1 / (a1 + a2)
    kappa = 0.45

    # calculate Mtot from a, P
    Mtot = (
        (4 * np.pi**2 * (a * u.au) ** 3 / (c.G * (P * u.day) ** 2))
        .to(u.M_sun)
        .value
    )

    M2 = kappa * Mtot
    M1 = Mtot - M2

    orbit = KeplerianOrbit(a=a * au_to_R_sun, period=P, m_planet=M2)

    assert np.allclose(M1, orbit.m_star.eval())
    assert np.allclose(M2, orbit.m_planet.eval())
    assert np.allclose(Mtot, orbit.m_total.eval())


def test_get_consistent_inputs():
    period0 = np.array([12.567, 45.132])
    r_star0 = 1.235
    m_star0 = 0.986
    m_planet0 = with_unit(np.array([1.543, 2.354]), u.M_earth)
    (
        a1,
        period1,
        rho_star1,
        r_star1,
        m_star1,
        m_planet1,
    ) = _get_consistent_inputs(
        None, period0, None, r_star0, m_star0, m_planet0
    )

    assert np.allclose(period0, period1.eval())
    assert np.allclose(r_star0, r_star1.eval())
    assert np.allclose(m_star0, m_star1.eval())
    assert np.allclose(
        m_planet0.eval(), m_planet1.eval() * u.M_sun.to(u.M_earth)
    )

    (
        a2,
        period2,
        rho_star2,
        r_star2,
        m_star2,
        m_planet2,
    ) = _get_consistent_inputs(a1, None, rho_star1, r_star0, None, m_planet1)
    assert np.allclose(period1.eval(), period2.eval())
    assert np.allclose(rho_star1.eval(), rho_star2.eval())
    assert np.allclose(r_star1.eval(), r_star2.eval())
    assert np.allclose(m_star1.eval(), m_star2.eval())
    assert np.allclose(m_planet1.eval(), m_planet2.eval())

    (
        a3,
        period3,
        rho_star3,
        r_star3,
        m_star3,
        m_planet3,
    ) = _get_consistent_inputs(a2, None, rho_star2, None, m_star2, m_planet2)
    assert np.allclose(period1.eval(), period3.eval())
    assert np.allclose(rho_star1.eval(), rho_star3.eval())
    assert np.allclose(r_star1.eval(), r_star3.eval())
    assert np.allclose(m_star1.eval(), m_star3.eval())
    assert np.allclose(m_planet1.eval(), m_planet3.eval())

    (
        a4,
        period4,
        rho_star4,
        r_star4,
        m_star4,
        m_planet4,
    ) = _get_consistent_inputs(a3, period3, None, r_star3, None, m_planet3)
    assert np.allclose(period1.eval(), period4.eval())
    assert np.allclose(rho_star1.eval(), rho_star4.eval())
    assert np.allclose(r_star1.eval(), r_star4.eval())
    assert np.allclose(m_star1.eval(), m_star4.eval())
    assert np.allclose(m_planet1.eval(), m_planet4.eval())

    (
        a5,
        period5,
        rho_star5,
        r_star5,
        m_star5,
        m_planet5,
    ) = _get_consistent_inputs(
        a3,
        None,
        with_unit(rho_star3, u.g / u.cm**3),
        r_star3,
        None,
        m_planet3,
    )
    assert np.allclose(period1.eval(), period5.eval())
    assert np.allclose(rho_star1.eval(), rho_star5.eval())
    assert np.allclose(r_star1.eval(), r_star5.eval())
    assert np.allclose(m_star1.eval(), m_star5.eval())
    assert np.allclose(m_planet1.eval(), m_planet5.eval())

    with pytest.raises(ValueError):
        _get_consistent_inputs(None, None, None, r_star3, m_star3, None)

    with pytest.raises(ValueError):
        _get_consistent_inputs(a3, period3, None, r_star3, m_star3, None)

    with pytest.raises(ValueError):
        _get_consistent_inputs(a3, None, rho_star3, r_star3, m_star3, None)


def test_light_delay():
    # Instantiate the orbit
    m_star = tt.scalar()
    period = tt.scalar()
    ecc = tt.scalar()
    omega = tt.scalar()
    Omega = tt.scalar()
    incl = tt.scalar()
    m_planet = tt.scalar()
    t = tt.scalar()
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.0,
        t0=0.0,
        period=period,
        ecc=ecc,
        omega=omega,
        Omega=Omega,
        incl=incl,
        m_planet=m_planet,
    )

    # True position
    get_position = theano.function(
        [t, m_star, period, ecc, omega, Omega, incl, m_planet],
        orbit.get_planet_position([t], light_delay=False),
    )

    # Retarded position
    get_retarded_position = theano.function(
        [t, m_star, period, ecc, omega, Omega, incl, m_planet],
        orbit.get_planet_position([t], light_delay=True),
    )

    # Retarded position (numerical)
    def get_exact_retarded_position(t, *args):
        def loss(params):
            (ti,) = params
            xr, yr, zr = get_position(ti, *args)
            delay = (zr * u.Rsun / c).to(u.day).value
            return (ti - delay - t) ** 2

        tr = minimize(loss, t).x[0]
        return get_position(tr, *args)

    # Compare for 100 different orbits
    np.random.seed(13)
    for i in range(100):
        m_star = 0.1 + np.random.random() * 1.9
        period = np.random.random() * 500
        ecc = np.random.random()
        omega = np.random.random() * 2 * np.pi
        Omega = np.random.random() * 2 * np.pi
        incl = np.random.random() * 0.5 * np.pi
        m_planet = np.random.random()
        t = np.random.random() * period
        args = (m_star, period, ecc, omega, Omega, incl, m_planet)
        assert np.allclose(
            np.reshape(get_retarded_position(t, *args), (-1,)),
            np.reshape(get_exact_retarded_position(t, *args), (-1,)),
        )


def test_light_delay_shape_two_planets_vector_t():
    orbit = KeplerianOrbit(period=[1.0, 2.0])
    t = np.linspace(0, 10, 50)
    x, y, z = orbit.get_planet_position(t, light_delay=False)
    xr, yr, zr = orbit.get_planet_position(t, light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_light_delay_shape_scalar_t():
    orbit = KeplerianOrbit(period=1.0)
    x, y, z = orbit.get_planet_position(1.0, light_delay=False)
    xr, yr, zr = orbit.get_planet_position(1.0, light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_light_delay_shape_single_t():
    orbit = KeplerianOrbit(period=1.0)
    x, y, z = orbit.get_planet_position([1.0], light_delay=False)
    xr, yr, zr = orbit.get_planet_position([1.0], light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_light_delay_shape_vector_t():
    orbit = KeplerianOrbit(period=1.0)
    x, y, z = orbit.get_planet_position([1.0, 2.0], light_delay=False)
    xr, yr, zr = orbit.get_planet_position([1.0, 2.0], light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_light_delay_shape_two_planets_scalar_t():
    orbit = KeplerianOrbit(period=[1.0, 2.0])
    x, y, z = orbit.get_planet_position(1.0, light_delay=False)
    xr, yr, zr = orbit.get_planet_position(1.0, light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_light_delay_shape_two_planets_single_t():
    orbit = KeplerianOrbit(period=[1.0, 2.0])
    x, y, z = orbit.get_planet_position([1.0], light_delay=False)
    xr, yr, zr = orbit.get_planet_position([1.0], light_delay=True)
    assert np.array_equal(x.shape.eval(), xr.shape.eval())


def test_get_aor_from_transit_duration():
    duration = 0.12
    period = 10.1235
    b = 0.34
    ror = 0.06
    r_star = 0.7

    dv = tt.as_tensor_variable(duration)
    aor, jac = get_aor_from_transit_duration(dv, period, b, ror)

    assert np.allclose(theano.grad(aor, dv).eval(), jac.eval())

    for orbit in [
        KeplerianOrbit(
            period=period, t0=0.0, b=b, a=r_star * aor, r_star=r_star
        ),
        KeplerianOrbit(
            period=period,
            t0=0.0,
            b=b,
            duration=duration,
            r_star=r_star,
            ror=ror,
        ),
    ]:
        x, y, z = orbit.get_planet_position(0.5 * duration)
        assert np.allclose(tt.sqrt(x**2 + y**2).eval(), r_star * (1 + ror))

        x, y, z = orbit.get_planet_position(-0.5 * duration)
        assert np.allclose(tt.sqrt(x**2 + y**2).eval(), r_star * (1 + ror))

        x, y, z = orbit.get_planet_position(period + 0.5 * duration)
        assert np.allclose(tt.sqrt(x**2 + y**2).eval(), r_star * (1 + ror))


@pytest.mark.filterwarnings("error::UserWarning")
def test_duration_without_ror_warning():
    duration = 0.12
    period = 10.1235
    b = 0.34
    ror = 0.06
    r_star = 0.7

    with pytest.raises(UserWarning):
        KeplerianOrbit(
            period=period, t0=0.0, b=b, duration=duration, r_star=r_star
        )

    KeplerianOrbit(
        period=period, t0=0.0, b=b, duration=duration, r_star=r_star, ror=ror
    )


def test_jacobians():
    duration = 0.12
    period = 10.1235
    b = 0.34
    ror = 0.06
    r_star = 0.7

    dv = tt.as_tensor_variable(duration)
    orbit = KeplerianOrbit(
        period=period, t0=0.0, b=b, duration=dv, r_star=r_star, ror=ror
    )
    assert np.allclose(
        orbit.jacobians["duration"]["a"].eval(),
        theano.grad(orbit.a, dv).eval(),
    )
    assert np.allclose(
        orbit.jacobians["duration"]["a_planet"].eval(),
        theano.grad(orbit.a_planet, dv).eval(),
    )
    assert np.allclose(
        orbit.jacobians["duration"]["a_star"].eval(),
        theano.grad(orbit.a_star, dv).eval(),
    )
    assert np.allclose(
        orbit.jacobians["duration"]["rho_star"].eval(),
        theano.grad(orbit.rho_star, dv).eval(),
    )

    bv = tt.as_tensor_variable(b)
    orbit = KeplerianOrbit(
        period=period, t0=0.0, b=bv, a=orbit.a, r_star=r_star, ror=ror
    )
    assert np.allclose(
        orbit.jacobians["b"]["cos_incl"].eval(),
        theano.grad(orbit.cos_incl, bv).eval(),
    )

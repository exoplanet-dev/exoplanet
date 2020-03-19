# -*- coding: utf-8 -*-

__all__ = [
    "KeplerianOrbit",
    "get_true_anomaly",
    "get_aor_from_transit_duration",
]

import warnings

import numpy as np
import theano.tensor as tt
from astropy import units as u
from theano.ifelse import ifelse

from ..citations import add_citations_to_model
from ..theano_ops.contact import ContactPointsOp
from ..theano_ops.kepler import KeplerOp
from ..units import has_unit, to_unit, with_unit
from .constants import G_grav, au_per_R_sun, gcc_per_sun


class KeplerianOrbit:
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
    5. Either ``t0`` (reference transit) or ``t_periastron`` must be given,
       but not both.


    Args:
        period: The orbital periods of the bodies in days.
        a: The semimajor axes of the orbits in ``R_sun``.
        t0: The time of a reference transit for each orbits in days.
        t_periastron: The epoch of a reference periastron passage in days.
        incl: The inclinations of the orbits in radians.
        b: The impact parameters of the orbits.
        ecc: The eccentricities of the orbits. Must be ``0 <= ecc < 1``.
        omega: The arguments of periastron for the orbits in radians.
        Omega: The position angles of the ascending nodes in radians.
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

    __citations__ = ("astropy",)

    def __init__(
        self,
        period=None,
        a=None,
        t0=None,
        t_periastron=None,
        incl=None,
        b=None,
        duration=None,
        ecc=None,
        omega=None,
        Omega=None,
        m_planet=0.0,
        m_star=None,
        r_star=None,
        rho_star=None,
        ror=None,
        m_planet_units=None,
        rho_star_units=None,
        model=None,
        contact_points_kwargs=None,
        **kwargs
    ):
        add_citations_to_model(self.__citations__, model=model)

        self.kepler_op = KeplerOp(**kwargs)

        if ecc is None and duration is not None:
            if r_star is None:
                r_star = tt.as_tensor_variable(1.0)
            if b is None:
                raise ValueError(
                    "'b' must be provided for a circular orbit with a "
                    "'duration'"
                )
            a = r_star * get_aor_from_transit_duration(
                duration, period, b, ror=ror
            )
            duration = None

        # Parameters
        if m_planet_units is not None:
            warnings.warn(
                "The `m_planet_units` argument has been deprecated. "
                "Use `with_unit` instead.",
                DeprecationWarning,
            )
            m_planet = with_unit(m_planet, m_planet_units)
        if rho_star_units is not None:
            warnings.warn(
                "The `rho_star_units` argument has been deprecated. "
                "Use `with_unit` instead.",
                DeprecationWarning,
            )
            rho_star = with_unit(rho_star, rho_star_units)
        inputs = _get_consistent_inputs(
            a, period, rho_star, r_star, m_star, m_planet
        )
        (
            self.a,
            self.period,
            self.rho_star,
            self.r_star,
            self.m_star,
            self.m_planet,
        ) = inputs
        self.m_total = self.m_star + self.m_planet

        self.n = 2 * np.pi / self.period
        self.a_star = self.a * self.m_planet / self.m_total
        self.a_planet = -self.a * self.m_star / self.m_total

        self.K0 = self.n * self.a / self.m_total

        # Set up the contact points calculation
        if contact_points_kwargs is None:
            contact_points_kwargs = dict()

        if Omega is None:
            self.Omega = None
        else:
            self.Omega = tt.as_tensor_variable(Omega)
            self.cos_Omega = tt.cos(self.Omega)
            self.sin_Omega = tt.sin(self.Omega)

        # Eccentricity
        self.contact_points_op = ContactPointsOp(**contact_points_kwargs)
        if ecc is None:
            self.ecc = None
            self.M0 = 0.5 * np.pi + tt.zeros_like(self.n)
            incl_factor = 1
        else:
            self.ecc = tt.as_tensor_variable(ecc)
            if omega is None:
                raise ValueError("both e and omega must be provided")
            self.omega = tt.as_tensor_variable(omega)

            self.cos_omega = tt.cos(self.omega)
            self.sin_omega = tt.sin(self.omega)

            opsw = 1 + self.sin_omega
            E0 = 2 * tt.arctan2(
                tt.sqrt(1 - self.ecc) * self.cos_omega,
                tt.sqrt(1 + self.ecc) * opsw,
            )
            self.M0 = E0 - self.ecc * tt.sin(E0)

            ome2 = 1 - self.ecc ** 2
            self.K0 /= tt.sqrt(ome2)
            incl_factor = (1 + self.ecc * self.sin_omega) / ome2

        # The Jacobian for the transform cos(i) -> b
        self.dcosidb = incl_factor * self.r_star / self.a

        if b is not None:
            if incl is not None or duration is not None:
                raise ValueError(
                    "only one of 'incl', 'b', and 'duration' can be given"
                )
            self.b = tt.as_tensor_variable(b)
            self.cos_incl = self.dcosidb * self.b
            self.incl = tt.arccos(self.cos_incl)
        elif incl is not None:
            if duration is not None:
                raise ValueError(
                    "only one of 'incl', 'b', and 'duration' can be given"
                )
            self.incl = tt.as_tensor_variable(incl)
            self.cos_incl = tt.cos(self.incl)
            self.b = self.cos_incl / self.dcosidb
        elif duration is not None:
            if self.ecc is None:
                raise ValueError(
                    "fitting with duration only works for eccentric orbits"
                )
            self.duration = tt.as_tensor_variable(to_unit(duration, u.day))
            c = tt.sin(np.pi * self.duration * incl_factor / self.period)
            c2 = c * c
            aor = self.a_planet / self.r_star
            esinw = self.ecc * self.sin_omega
            self.b = tt.sqrt(
                (aor ** 2 * c2 - 1)
                / (
                    c2 * esinw ** 2
                    + 2 * c2 * esinw
                    + c2
                    - self.ecc ** 4
                    + 2 * self.ecc ** 2
                    - 1
                )
            )
            self.b *= 1 - self.ecc ** 2
            self.cos_incl = self.dcosidb * self.b
            self.incl = tt.arccos(self.cos_incl)
        else:
            zla = tt.zeros_like(self.a)
            self.incl = 0.5 * np.pi + zla
            self.cos_incl = zla
            self.b = zla

        if t0 is not None and t_periastron is not None:
            raise ValueError("you can't define both t0 and t_periastron")
        if t0 is None and t_periastron is None:
            t0 = tt.zeros_like(self.period)

        if t0 is None:
            self.t_periastron = tt.as_tensor_variable(t_periastron)
            self.t0 = self.t_periastron + self.M0 / self.n
        else:
            self.t0 = tt.as_tensor_variable(t0)
            self.t_periastron = self.t0 - self.M0 / self.n

        self.tref = self.t_periastron - self.t0

        self.sin_incl = tt.sin(self.incl)

    def _rotate_vector(self, x, y):
        """Apply the rotation matrices to go from orbit to observer frame

        In order,
        1. rotate about the z axis by an amount omega -> x1, y1, z1
        2. rotate about the x1 axis by an amount -incl -> x2, y2, z2
        3. rotate about the z2 axis by an amount Omega -> x3, y3, z3

        Args:
            x: A tensor representing the x coodinate in the plane of the
                orbit.
            y: A tensor representing the y coodinate in the plane of the
                orbit.

        Returns:
            Three tensors representing ``(X, Y, Z)`` in the observer frame.

        """

        # 1) rotate about z0 axis by omega
        if self.ecc is None:
            x1 = x
            y1 = y
        else:
            x1 = self.cos_omega * x - self.sin_omega * y
            y1 = self.sin_omega * x + self.cos_omega * y

        # 2) rotate about x1 axis by -incl
        x2 = x1
        y2 = self.cos_incl * y1
        # z3 = z2, subsequent rotation by Omega doesn't affect it
        Z = -self.sin_incl * y1

        # 3) rotate about z2 axis by Omega
        if self.Omega is None:
            return (x2, y2, Z)

        X = self.cos_Omega * x2 - self.sin_Omega * y2
        Y = self.sin_Omega * x2 + self.cos_Omega * y2
        return X, Y, Z

    def _warp_times(self, t):
        return tt.shape_padright(t) - self.t0

    def _get_true_anomaly(self, t):
        M = (self._warp_times(t) - self.tref) * self.n
        if self.ecc is None:
            return tt.sin(M), tt.cos(M)
        sinf, cosf = self.kepler_op(M, self.ecc + tt.zeros_like(M))
        return sinf, cosf

    def _get_position_and_velocity(self, t, parallax=None):
        sinf, cosf = self._get_true_anomaly(t)

        if self.ecc is None:
            r = 1.0
            vx, vy, vz = self._rotate_vector(-self.K0 * sinf, self.K0 * cosf)
        else:
            r = (1.0 - self.ecc ** 2) / (1 + self.ecc * cosf)
            vx, vy, vz = self._rotate_vector(
                -self.K0 * sinf, self.K0 * (cosf + self.ecc)
            )

        x, y, z = self._rotate_vector(r * cosf, r * sinf)

        pos = tt.stack((x, y, z), axis=-1)
        pos = tt.concatenate(
            (
                tt.sum(
                    tt.shape_padright(self.a_star) * pos, axis=0, keepdims=True
                ),
                tt.shape_padright(self.a_planet) * pos,
            ),
            axis=0,
        )
        vel = tt.stack((vx, vy, vz), axis=-1)
        vel = tt.concatenate(
            (
                tt.sum(
                    tt.shape_padright(self.m_planet) * vel,
                    axis=0,
                    keepdims=True,
                ),
                -tt.shape_padright(self.m_star) * vel,
            ),
            axis=0,
        )

        if parallax is not None:
            # convert r into arcseconds
            pos = pos * (parallax * au_per_R_sun)
            vel = vel * (parallax * au_per_R_sun)

        return pos, vel

    def _get_position(self, a, t, parallax=None):
        """Get the position of a body.

        Args:
            a: the semi-major axis of the orbit.
            t: the time (or tensor of times) to calculate the position.
            parallax: (arcseconds) if provided, return the position in
            units of arcseconds.

        Returns:
            The position of the body in the observer frame. Default is in units
            of R_sun, but if parallax is provided, then in units of arcseconds.

        """
        sinf, cosf = self._get_true_anomaly(t)
        if self.ecc is None:
            r = a
        else:
            r = a * (1.0 - self.ecc ** 2) / (1 + self.ecc * cosf)

        if parallax is not None:
            # convert r into arcseconds
            r = r * parallax * au_per_R_sun

        return self._rotate_vector(r * cosf, r * sinf)

    def get_planet_position(self, t, parallax=None):
        """The planets' positions in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        return tuple(
            tt.squeeze(x)
            for x in self._get_position(self.a_planet, t, parallax)
        )

    def get_star_position(self, t, parallax=None):
        """The star's position in the barycentric frame

        .. note:: If there are multiple planets in the system, this will
            return one column per planet with each planet's contribution to
            the motion. The star's full position can be computed by summing
            over the last axis.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        return tuple(
            tt.squeeze(x) for x in self._get_position(self.a_star, t, parallax)
        )

    def get_relative_position(self, t, parallax=None):
        """The planets' positions relative to the star in the X,Y,Z frame.

        .. note:: This treats each planet independently and does not take the
            other planets into account when computing the position of the
            star. This is fine as long as the planet masses are small. In
            other words, the reflex motion of the star caused by the other
            planets is neglected when computing the relative coordinates of
            one of the planets.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.

        """
        return tuple(
            tt.squeeze(x) for x in self._get_position(-self.a, t, parallax)
        )

    def get_relative_angles(self, t, parallax=None):
        """The planets' relative position to the star in the sky plane, in
        separation, position angle coordinates.

        .. note:: This treats each planet independently and does not take the
            other planets into account when computing the position of the
            star. This is fine as long as the planet masses are small.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The separation (arcseconds) and position angle (radians,
            measured east of north) of the planet relative to the star.

        """

        X, Y, Z = self._get_position(-self.a, t, parallax)

        # calculate rho and theta
        rho = tt.squeeze(tt.sqrt(X ** 2 + Y ** 2))  # arcsec
        theta = tt.squeeze(tt.arctan2(Y, X))  # radians between [-pi, pi]

        return (rho, theta)

    def _get_velocity(self, m, t):
        """Get the velocity vector of a body in the observer frame"""
        sinf, cosf = self._get_true_anomaly(t)
        K = self.K0 * m
        if self.ecc is None:
            return self._rotate_vector(-K * sinf, K * cosf)
        return self._rotate_vector(-K * sinf, K * (cosf + self.ecc))

    def get_planet_velocity(self, t):
        """Get the planets' velocity vectors

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.

        """
        return tuple(
            tt.squeeze(x) for x in self._get_velocity(-self.m_star, t)
        )

    def get_star_velocity(self, t):
        """Get the star's velocity vector

        .. note:: For a system with multiple planets, this will return one
            column per planet with the contributions from each planet. The
            total velocity can be found by summing along the last axis.

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.

        """
        return tuple(
            tt.squeeze(x) for x in self._get_velocity(self.m_planet, t)
        )

    def get_relative_velocity(self, t):
        """The planets' velocity relative to the star

        .. note:: This treats each planet independently and does not take the
            other planets into account when computing the position of the
            star. This is fine as long as the planet masses are small.

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``R_sun/day``.

        """
        return tuple(
            tt.squeeze(x) for x in self._get_velocity(-self.m_total, t)
        )

    def get_radial_velocity(self, t, K=None, output_units=None):
        """Get the radial velocity of the star

        .. note:: The convention in exoplanet is that positive `z` points
            *towards* the observer. However, for consistency with radial
            velocity literature this method returns values where positive
            radial velocity corresponds to a redshift as expected.

        Args:
            t: The times where the radial velocity should be evaluated.
            K (Optional): The semi-amplitudes of the orbits. If provided, the
                ``m_planet`` and ``incl`` parameters will be ignored and this
                amplitude will be used instead.
            output_units (Optional): An AstroPy velocity unit. If not given,
                the output will be evaluated in ``m/s``. This is ignored if a
                value is given for ``K``.

        Returns:
            The reflex radial velocity evaluated at ``t`` in units of
            ``output_units``. For multiple planets, this will have one row for
            each planet.

        """

        # Special case for K given: m_planet, incl, etc. is ignored
        if K is not None:
            sinf, cosf = self._get_true_anomaly(t)
            if self.ecc is None:
                return tt.squeeze(K * cosf)
            # cos(w + f) + e * cos(w) from Lovis & Fischer
            return tt.squeeze(
                K
                * (
                    self.cos_omega * cosf
                    - self.sin_omega * sinf
                    + self.ecc * self.cos_omega
                )
            )

        # Compute the velocity using the full orbit solution
        if output_units is None:
            output_units = u.m / u.s
        conv = (1 * u.R_sun / u.day).to(output_units).value
        v = self.get_star_velocity(t)
        return -conv * v[2]

    def _get_acceleration(self, a, m, t):
        sinf, cosf = self._get_true_anomaly(t)
        K = self.K0 * m
        if self.ecc is None:
            factor = -(K ** 2) / a
        else:
            factor = (
                K ** 2 * (self.ecc * cosf + 1) ** 2 / (a * (self.ecc ** 2 - 1))
            )
        return self._rotate_vector(factor * cosf, factor * sinf)

    def get_planet_acceleration(self, t):
        return tuple(
            tt.squeeze(x)
            for x in self._get_acceleration(self.a_planet, -self.m_star, t)
        )

    def get_star_acceleration(self, t):
        return tuple(
            tt.squeeze(x)
            for x in self._get_acceleration(self.a_star, self.m_planet, t)
        )

    def get_relative_acceleration(self, t):
        return tuple(
            tt.squeeze(x)
            for x in self._get_acceleration(-self.a, -self.m_total, t)
        )

    def in_transit(self, t, r=0.0, texp=None):
        """Get a list of timestamps that are in transit

        Args:
            t (vector): A vector of timestamps to be evaluated.
            r (Optional): The radii of the planets.
            texp (Optional[float]): The exposure time.

        Returns:
            The indices of the timestamps that are in transit.

        """
        z = tt.zeros_like(self.a)
        r = tt.as_tensor_variable(r) + z
        R = self.r_star + z

        # Wrap the times into time since transit
        hp = 0.5 * self.period
        dt = tt.mod(self._warp_times(t) + hp, self.period) - hp

        if self.ecc is None:
            # Equation 14 from Winn (2010)
            k = r / R
            arg = tt.square(1 + k) - tt.square(self.b)
            factor = R / (self.a * self.sin_incl)
            hdur = hp * tt.arcsin(factor * tt.sqrt(arg)) / np.pi
            t_start = -hdur
            t_end = hdur
            flag = z

        else:
            M_contact = self.contact_points_op(
                self.a,
                self.ecc,
                self.cos_omega,
                self.sin_omega,
                self.cos_incl + z,
                self.sin_incl + z,
                R + r,
            )
            flag = M_contact[2]

            t_start = (M_contact[0] - self.M0) / self.n
            t_start = tt.mod(t_start + hp, self.period) - hp
            t_end = (M_contact[1] - self.M0) / self.n
            t_end = tt.mod(t_end + hp, self.period) - hp

            t_start = tt.switch(
                tt.gt(t_start, 0.0), t_start - self.period, t_start
            )
            t_end = tt.switch(tt.lt(t_end, 0.0), t_end + self.period, t_end)

        if texp is not None:
            t_start -= 0.5 * texp
            t_end += 0.5 * texp

        mask = tt.any(tt.and_(dt >= t_start, dt <= t_end), axis=-1)
        result = ifelse(
            tt.all(tt.eq(flag, 0)), tt.arange(t.size)[mask], tt.arange(t.size)
        )

        return result

    def _flip(self, r_planet, model=None):
        orbit = type(self)(
            period=self.period,
            t_periastron=self.t_periastron,
            incl=self.incl,
            ecc=self.ecc,
            omega=self.omega - np.pi,
            Omega=self.Omega,
            m_star=self.m_planet,
            m_planet=self.m_star,
            r_star=r_planet,
            model=model,
        )
        orbit.kepler_op = self.kepler_op
        orbit.contact_points_op = self.contact_points_op
        return orbit


def get_true_anomaly(M, e, **kwargs):
    """Get the true anomaly for a tensor of mean anomalies and eccentricities

    Args:
        M: The mean anomaly.
        e: The eccentricity. This should have the same shape as ``M``.

    Returns:
        The true anomaly of the orbit.

    """
    sinf, cosf = KeplerOp()(M, e)
    return tt.arctan2(sinf, cosf)


def get_aor_from_transit_duration(duration, period, b, ror=None):
    """Get the semimajor axis implied by a circular orbit and duration

    Args:
        duration: The transit duration
        period: The orbital period
        b: The impact parameter of the transit
        ror: The radius ratio of the planet to the star

    Returns:
        The semimajor axis in units of the stellar radius

    """
    if ror is None:
        ror = tt.as_tensor_variable(0.0)
    sin2_phi = tt.sin(np.pi * duration / period) ** 2
    return tt.sqrt(((1 + ror) ** 2 - b ** 2 * (1 - sin2_phi)) / sin2_phi)


def _get_consistent_inputs(a, period, rho_star, r_star, m_star, m_planet):
    if a is None and period is None:
        raise ValueError(
            "values must be provided for at least one of a " "and period"
        )

    if m_planet is not None:
        m_planet = tt.as_tensor_variable(to_unit(m_planet, u.M_sun))

    if a is not None:
        a = tt.as_tensor_variable(to_unit(a, u.R_sun))
        if m_planet is None:
            m_planet = tt.zeros_like(a)
    if period is not None:
        period = tt.as_tensor_variable(to_unit(period, u.day))
        if m_planet is None:
            m_planet = tt.zeros_like(period)

    # Compute the implied density if a and period are given
    implied_rho_star = False
    if a is not None and period is not None:
        if rho_star is not None or m_star is not None:
            raise ValueError(
                "if both a and period are given, you can't "
                "also define rho_star or m_star"
            )

        # Default to a stellar radius of 1 if not provided
        if r_star is None:
            r_star = tt.as_tensor_variable(1.0)
        else:
            r_star = tt.as_tensor_variable(to_unit(r_star, u.R_sun))

        # Compute the implied mass via Kepler's 3rd law
        m_tot = 4 * np.pi * np.pi * a ** 3 / (G_grav * period ** 2)

        # Compute the implied density
        m_star = m_tot - m_planet
        vol_star = 4 * np.pi * r_star ** 3 / 3.0
        rho_star = m_star / vol_star
        implied_rho_star = True

    # Make sure that the right combination of stellar parameters are given
    if r_star is None and m_star is None:
        r_star = 1.0
        if rho_star is None:
            m_star = 1.0
    if (not implied_rho_star) and sum(
        arg is None for arg in (rho_star, r_star, m_star)
    ) != 1:
        raise ValueError(
            "values must be provided for exactly two of "
            "rho_star, m_star, and r_star"
        )

    if rho_star is not None and not implied_rho_star:
        if has_unit(rho_star):
            rho_star = tt.as_tensor_variable(
                to_unit(rho_star, u.M_sun / u.R_sun ** 3)
            )
        else:
            rho_star = tt.as_tensor_variable(rho_star) / gcc_per_sun
    if r_star is not None:
        r_star = tt.as_tensor_variable(to_unit(r_star, u.R_sun))
    if m_star is not None:
        m_star = tt.as_tensor_variable(to_unit(m_star, u.M_sun))

    # Work out the stellar parameters
    if rho_star is None:
        rho_star = 3 * m_star / (4 * np.pi * r_star ** 3)
    elif r_star is None:
        r_star = (3 * m_star / (4 * np.pi * rho_star)) ** (1 / 3)
    elif m_star is None:
        m_star = 4 * np.pi * r_star ** 3 * rho_star / 3.0

    # Work out the planet parameters
    if a is None:
        a = (
            G_grav * (m_star + m_planet) * period ** 2 / (4 * np.pi ** 2)
        ) ** (1.0 / 3)
    elif period is None:
        period = (
            2 * np.pi * a ** (3 / 2) / (tt.sqrt(G_grav * (m_star + m_planet)))
        )

    return a, period, rho_star * gcc_per_sun, r_star, m_star, m_planet

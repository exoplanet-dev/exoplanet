# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["estimate_semi_amplitude", "estimate_minimum_mass",
           "lomb_scargle_estimator", "autocorr_estimator"]

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

import astropy.units as u
from astropy.stats import LombScargle


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


def estimate_semi_amplitude(periods, x, y, yerr=None, t0s=None):
    """Estimate the RV semi-amplitudes for planets in an RV series

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

    Returns:
        An estimate of the semi-amplitude of each planet in units of ``m/s``.

    """
    if yerr is None:
        ivar = np.ones_like(y)
    else:
        ivar = 1.0 / yerr**2

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
    return K


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
        An estimate of the minimum mass of each planet as an AstroPy Quantity
        with units of ``M_jupiter``.

    """
    periods = u.Quantity(np.atleast_1d(periods), unit=u.day)
    m_star = u.Quantity(m_star, unit=u.M_sun)
    K = estimate_semi_amplitude(periods, x, y, yerr=yerr, t0s=t0s)
    m_J = K / 28.4329 * m_star.value**(2./3)
    m_J *= (periods.to(u.year)).value**(1./3)
    return m_J * u.M_jupiter


def lomb_scargle_estimator(x, y, yerr=None,
                           min_period=None, max_period=None,
                           filter_period=None,
                           max_peaks=2,
                           **kwargs):
    """Estimate period of a time series using the periodogram

    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        filter_period (Optional[float]): If given, use a high-pass filter to
            down-weight period longer than this
        max_peaks (Optional[int]): The maximum number of peaks to return
            (default: 2)

    Returns:
        A dictionary with the computed ``periodogram`` and the parameters for
        up to ``max_peaks`` peaks in the periodogram.

    """
    if min_period is not None:
        kwargs["maximum_frequency"] = 1.0 / min_period
    if max_period is not None:
        kwargs["minimum_frequency"] = 1.0 / max_period

    # Estimate the power spectrum
    model = LombScargle(x, y, yerr)
    freq, power = model.autopower(method="fast", normalization="psd", **kwargs)
    power /= len(x)
    power_est = np.array(power)

    # Filter long periods
    if filter_period is not None:
        freq0 = 1.0 / filter_period
        filt = 1.0 / np.sqrt(1 + (freq0 / freq) ** (2*3))
        power *= filt

    # Find and fit peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for i in peak_inds[:max_peaks]:
        A = np.vander(freq[i-1:i+2], 3)
        w = np.linalg.solve(A, np.log(power[i-1:i+2]))
        sigma2 = -0.5 / w[0]
        freq0 = w[1] * sigma2
        peaks.append(dict(
            log_power=w[2] + 0.5*freq0**2 / sigma2,
            period=1.0 / freq0,
            period_uncert=np.sqrt(sigma2 / freq0**4),
        ))

    return dict(
        periodogram=(freq, power_est),
        peaks=peaks,
    )


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_function(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    .. note:: This is from `emcee <https://github.com/dfm/emcee>`_.

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        The autocorrelation function of the time series.

    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def autocorr_estimator(x, y, yerr=None,
                       min_period=None, max_period=None,
                       oversample=2.0, smooth=2.0, max_peaks=10):
    """Estimate the period of a time series using the autocorrelation function

    .. note:: The signal is interpolated onto a uniform grid in time so that
        the autocorrelation function can be computed.

    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        oversample (Optional[float]): When interpolating, oversample the times
            by this factor (default: 2.0)
        smooth (Optional[float]): Smooth the autocorrelation function by this
            factor times the minimum period (default: 2.0)
        max_peaks (Optional[int]): The maximum number of peaks to identify in
            the autocorrelation function (default: 10)

    Returns:
        A dictionary with the computed autocorrelation function and the
        estimated period. For compatibility with the
        :func:`lomb_scargle_estimator`, the period is returned as a list with
        the key ``peaks``.

    """
    if gaussian_filter is None:
        raise ImportError("scipy is required to use the autocorr estimator")

    if min_period is None:
        min_period = np.min(np.diff(x))
    if max_period is None:
        max_period = x.max() - x.min()

    # Interpolate onto an evenly spaced grid
    dx = np.min(np.diff(x)) / float(oversample)
    xx = np.arange(x.min(), x.max(), dx)
    yy = np.interp(xx, x, y)

    # Estimate the autocorrelation function
    tau = xx - x[0]
    acor = autocorr_function(yy)
    smooth = smooth * min_period
    acor = gaussian_filter(acor, smooth / dx)

    # Find the peaks
    peak_inds = (acor[1:-1] > acor[:-2]) & (acor[1:-1] > acor[2:])
    peak_inds = np.arange(1, len(acor)-1)[peak_inds]
    peak_inds = peak_inds[tau[peak_inds] >= min_period]

    result = dict(
        autocorr=(tau, acor),
        peaks=[],
    )

    # No peaks were found
    if len(peak_inds) == 0 or tau[peak_inds[0]] > max_period:
        return result

    # Only one peak was found
    if len(peak_inds) == 1:
        result["peaks"] = [dict(period=tau[peak_inds[0]],
                                period_uncert=np.nan)]
        return result

    # Check to see if second peak is higher
    if acor[peak_inds[1]] > acor[peak_inds[0]]:
        peak_inds = peak_inds[1:]

    # The first peak is larger than the maximum period
    if tau[peak_inds[0]] > max_period:
        return result

    result["peaks"] = [dict(period=tau[peak_inds[0]], period_uncert=np.nan)]
    return result

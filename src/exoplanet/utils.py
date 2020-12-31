# -*- coding: utf-8 -*-

__all__ = [
    "logger",
    "deprecation_warning",
    "deprecated",
    "compute_expected_transit_times",
]

import logging
import warnings
from functools import wraps

import numpy as np

logger = logging.getLogger("exoplanet")


def deprecation_warning(msg):
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def deprecated(alternate=None):
    def wrapper(func, alternate=alternate):
        msg = "'{0}' is deprecated.".format(func.__name__)
        if alternate is not None:
            msg += " Use '{0}' instead.".format(alternate)

        @wraps(func)
        def f(*args, **kwargs):
            deprecation_warning(msg)
            return func(*args, **kwargs)

        return f

    return wrapper


def compute_expected_transit_times(min_time, max_time, period, t0):
    """Compute the expected transit times within a dataset

    Args:
        min_time (float): The start time of the dataset
        max_time (float): The end time of the dataset
        period (array): The periods of the planets
        t0 (array): The reference transit times for the planets

    Returns:
        A list of arrays of expected transit times for each planet

    """
    periods = np.atleast_1d(period)
    t0s = np.atleast_1d(t0)
    transit_times = []
    for period, t0 in zip(periods, t0s):
        min_ind = np.floor((min_time - t0) / period)
        max_ind = np.ceil((max_time - t0) / period)
        times = t0 + period * np.arange(min_ind, max_ind, 1)
        times = times[(min_time <= times) & (times <= max_time)]
        transit_times.append(times)
    return transit_times

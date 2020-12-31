# -*- coding: utf-8 -*-
__all__ = ["light_curves", "LimbDarkLightCurve", "orbits", "units"]

import warnings

import theano

from . import light_curves, orbits, units
from .light_curves import LimbDarkLightCurve

if theano.config.floatX != "float64":
    warnings.warn(
        "exoplanet should only be used with 'float64' precision, "
        "but theano.config.floatX == '{0}'".format(theano.config.floatX)
    )

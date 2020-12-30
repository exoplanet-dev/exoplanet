# -*- coding: utf-8 -*-
__all__ = ["orbits", "light_curves", "KeplerianOrbit", "LimbDarkLightCurve"]


import warnings

import theano

from . import light_curves, orbits
from .light_curves import LimbDarkLightCurve
from .orbits import KeplerianOrbit

if theano.config.floatX != "float64":
    warnings.warn(
        "exoplanet should only be used with 'float64' precision, "
        "but theano.config.floatX == '{0}'".format(theano.config.floatX)
    )

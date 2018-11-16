# -*- coding: utf-8 -*-

__version__ = "0.0.1"

try:
    __EXOPLANET_SETUP__
except NameError:
    __EXOPLANET_SETUP__ = False

if not __EXOPLANET_SETUP__:
    __all__ = ["distributions", "gp", "orbits", "utils",
               "StarryLightCurve"]

    from . import distributions, gp, orbits, utils
    from .light_curve import StarryLightCurve

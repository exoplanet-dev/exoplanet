# -*- coding: utf-8 -*-

from .exoplanet_version import __version__  # NOQA

try:
    __EXOPLANET_SETUP__
except NameError:
    __EXOPLANET_SETUP__ = False

if not __EXOPLANET_SETUP__:
    __all__ = ["distributions", "gp", "orbits"]

    from .utils import *  # NOQA
    from .sampling import *  # NOQA
    from .estimators import *  # NOQA
    from .light_curves import *  # NOQA

    from . import distributions, gp, orbits

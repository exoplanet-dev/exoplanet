# -*- coding: utf-8 -*-

from .exoplanet_version import __version__  # NOQA

try:
    __EXOPLANET_SETUP__
except NameError:
    __EXOPLANET_SETUP__ = False

if not __EXOPLANET_SETUP__:
    __all__ = [
        "distributions",
        "gp",
        "orbits",
        "interp",
        "get_dense_nuts_step",
    ]

    from .utils import *  # NOQA
    from .sampling import *  # NOQA
    from .estimators import *  # NOQA
    from .light_curves import *  # NOQA
    from .quadpotential import get_dense_nuts_step

    from . import distributions, gp, orbits, interp

    from .citations import CITATIONS

    __bibtex__ = __citation__ = CITATIONS["exoplanet"][1]

# -*- coding: utf-8 -*-

__all__ = ["terms", "GP"]

from celerite2.theano import GaussianProcess as GP
from celerite2.theano import terms

from .utils import deprecation_warning

deprecation_warning(
    "The exoplanet.gp submodule is deprecated. Use 'celerite2' instead."
)

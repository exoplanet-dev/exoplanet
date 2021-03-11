# -*- coding: utf-8 -*-

__all__ = ["optimize"]


from pymc3_ext import optimize

from .utils import deprecated

optimize = deprecated("pymc3_ext.optimize")(optimize)

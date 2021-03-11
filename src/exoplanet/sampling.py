# -*- coding: utf-8 -*-

__all__ = ["sample"]

from pymc3_ext import sample

from .utils import deprecated

sample = deprecated("pymc3_ext.sample")(sample)

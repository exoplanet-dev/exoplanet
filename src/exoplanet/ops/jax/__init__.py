# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

from jax.config import config  # noqa isort:skip

if not config.read("jax_enable_x64"):
    logger.warning(
        "The JAX interface for exoplanet only works with dtype float64. "
        "We're enabling x64 now, but you might run into issues if you've "
        "already run some jax code.\n"
        "You can squash this warning by setting the environment variable "
        "'JAX_ENABLE_X64=True' or by running:\n"
        ">>> from jax.config import config\n"
        ">>> config.update('jax_enable_x64', True)"
    )
    config.update("jax_enable_x64", True)


__all__ = ["ops"]

from . import ops  # noqa isort:skip

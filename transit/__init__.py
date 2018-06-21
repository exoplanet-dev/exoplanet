# -*- coding: utf-8 -*-

from __future__ import division, print_function

# __all__ = ["kepler"]

import os
import sysconfig
import tensorflow as tf


# Load the ops library
suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
libfile = os.path.join(dirname, "transit_op")
if suffix is not None:
    libfile += suffix
else:
    libfile += ".so"
custom_ops = tf.load_op_library(libfile)

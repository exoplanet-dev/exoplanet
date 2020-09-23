# -*- coding: utf-8 -*-

__all__ = ["resize_or_set"]

import numpy as np


def resize_or_set(outputs, n, shape, dtype=np.float64):
    if outputs[n][0] is None:
        outputs[n][0] = np.empty(shape, dtype=dtype)
    else:
        outputs[n][0] = np.ascontiguousarray(
            np.resize(outputs[n][0], shape), dtype=dtype
        )
    return outputs[n][0]

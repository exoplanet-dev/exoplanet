# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pymc3 as pm

from .sampling import PyMC3Sampler


def test_tuning():
    ndim = 5
    np.random.seed(42)

    sampler = PyMC3Sampler(start=10, window=50, finish=10)
    with pm.Model() as model:
        pm.Normal("x", sd=np.exp(np.random.uniform(-10, 0, ndim)), shape=ndim)

        trace = sampler.tune(tune=200, progressbar=False)

        start, step = sampler._get_start_and_step(
            start=None, step=None)

        # Expected step size
        expected = []
        for chain in trace._straces.values():
            expected.append(chain.get_sampler_stats("step_size")[-1])
        expected = np.mean(expected)
        assert np.allclose(step.step_size, expected)

        # Make sure that it's not still the default
        assert not np.allclose(step.step_size, 0.25 / model.ndim ** 0.25)

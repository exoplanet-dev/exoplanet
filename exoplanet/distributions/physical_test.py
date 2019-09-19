# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm
from scipy.stats import kstest

from .base_test import _Base
from .physical import ImpactParameter, QuadLimbDark


class TestPhysical(_Base):
    random_seed = 19860925

    def test_quad_limb_dark(self):
        with self._model():
            dist = QuadLimbDark("u", shape=2)

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 2)

            logp = QuadLimbDark.dist(shape=2).logp(samples).eval().flatten()
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        u1 = trace["u"][:, 0]
        u2 = trace["u"][:, 1]

        # Make sure that the physical constraints are satisfied
        assert np.all(u1 + u2 < 1)
        assert np.all(u1 > 0)
        assert np.all(u1 + 2 * u2 > 0)

        # Make sure that the qs are uniform
        q1 = (u1 + u2) ** 2
        q2 = 0.5 * u1 / (u1 + u2)

        cdf = lambda x: np.clip(x, 0, 1)  # NOQA
        for q in (q1, q2):
            s, p = kstest(q, cdf)
            assert s < 0.05

    def test_impact(self):
        lower = 0.1
        upper = 1.0
        with self._model():
            ror = pm.Uniform("ror", lower=lower, upper=upper, shape=(5, 2))
            dist = ImpactParameter("b", ror=ror)

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)
            assert np.all((0 <= samples) & (samples <= 1 + upper))

            trace = self._sample()

        u = trace["ror"]
        u = np.reshape(u, (len(u), -1))
        cdf = lambda x: np.clip((x - lower) / (upper - lower), 0, 1)  # NOQA
        for i in range(u.shape[1]):
            s, p = kstest(u[:, i], cdf)
            assert s < 0.05

        assert np.all(trace["b"] <= 1 + trace["ror"])

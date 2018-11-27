# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np
from scipy.stats import kstest

import pymc3 as pm

from .distributions import UnitVector, Angle, QuadLimbDark, RadiusImpact


class TestDistributions(object):
    random_seed = 20160911

    @classmethod
    def setup_class(cls):
        np.random.seed(cls.random_seed)

    @classmethod
    def teardown_class(cls):
        pm.theanof.set_theano_conf({"compute_test_value": "off"})

    def setup_method(self):
        np.random.seed(self.random_seed)

    def teardown_method(self, method):
        pm.theanof.set_theano_conf({"compute_test_value": "off"})

    def _sample(self, **kwargs):
        logger = logging.getLogger("pymc3")
        logger.propagate = False
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        return pm.sample(**kwargs)

    def _model(self, **kwargs):
        return pm.Model(**kwargs)

    def test_unit_vector(self):
        with self._model():
            UnitVector("x", shape=(2, 3))
            trace = self._sample()

        # Make sure that the unit vector constraint is satisfied
        assert np.allclose(np.sum(trace["x"]**2, axis=-1), 1.0)

        # Pull out the component and compute the angle
        x = trace["x"][:, :, 0]
        y = trace["x"][:, :, 1]
        z = trace["x"][:, :, 2]
        theta = np.arctan2(y, x)

        # The angle should be uniformly distributed
        cdf = lambda x: np.clip((x + np.pi) / (2 * np.pi), 0, 1)  # NOQA
        for i in range(theta.shape[1]):
            s, p = kstest(theta[:, i], cdf)
            assert s < 0.05

        # As should the vertical component
        cdf = lambda x: np.clip((x + 1) / 2, 0, 1)  # NOQA
        for i in range(z.shape[1]):
            s, p = kstest(z[:, i], cdf)
            assert s < 0.05

    def test_angle(self):
        with self._model():
            Angle("theta", shape=(5, 2))
            trace = self._sample()

        # The angle should be uniformly distributed
        theta = trace["theta"]
        theta = np.reshape(theta, (len(theta), -1))
        cdf = lambda x: np.clip((x + np.pi) / (2 * np.pi), 0, 1)  # NOQA
        for i in range(theta.shape[1]):
            s, p = kstest(theta[:, i], cdf)
            assert s < 0.05

    def test_quad_limb_dark(self):
        with self._model():
            QuadLimbDark("u", shape=2)
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

    def test_radius_impact(self):
        min_radius = 0.01
        max_radius = 0.1
        with self._model():
            RadiusImpact("rb", min_radius=min_radius, max_radius=max_radius)
            trace = self._sample()

        r = trace["rb"][:, 0]
        b = trace["rb"][:, 1]

        # Make sure that the physical constraints are satisfied
        assert np.all((r <= max_radius) & (min_radius <= r))
        assert np.all((b >= 0) & (b <= 1 + r))

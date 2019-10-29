# -*- coding: utf-8 -*-

import logging

import numpy as np
import pymc3 as pm
import pytest
from scipy.stats import kstest

from .base import Angle, Periodic, UnitUniform, UnitVector


class _Base:
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
        logger.setLevel(logging.ERROR)
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        return pm.sample(**kwargs)

    def _model(self, **kwargs):
        return pm.Model(**kwargs)


class TestBase(_Base):
    def test_unit_uniform(self):
        with self._model():
            dist = UnitUniform("u", shape=(5, 2))

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)
            assert np.all((0 <= samples) & (samples <= 1))

            logp = (
                UnitUniform.dist(shape=(5, 2)).logp(samples).eval().flatten()
            )
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        u = trace["u"]
        u = np.reshape(u, (len(u), -1))
        cdf = lambda x: np.clip(x, 0, 1)  # NOQA
        for i in range(u.shape[1]):
            s, p = kstest(u[:, i], cdf)
            assert s < 0.05

    def test_unit_vector(self):
        with self._model():
            dist = UnitVector("x", shape=(2, 3))

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 2, 3)
            assert np.allclose(np.sum(samples ** 2, axis=-1), 1.0)

            logp = np.sum(
                UnitVector.dist(shape=(2, 3)).logp(samples).eval(), axis=-1
            ).flatten()
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        # Make sure that the unit vector constraint is satisfied
        assert np.allclose(np.sum(trace["x"] ** 2, axis=-1), 1.0)

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

    @pytest.mark.parametrize("regularized", [None, 10.0])
    def test_angle(self, regularized):
        with self._model():
            dist = Angle("theta", shape=(5, 2), regularized=regularized)

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)
            assert np.all((-np.pi <= samples) & (samples <= np.pi))

            logp = Angle.dist(shape=(5, 2)).logp(samples).eval().flatten()
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        # The angle should be uniformly distributed
        theta = trace["theta"]
        theta = np.reshape(theta, (len(theta), -1))
        cdf = lambda x: np.clip((x + np.pi) / (2 * np.pi), 0, 1)  # NOQA
        for i in range(theta.shape[1]):
            s, p = kstest(theta[:, i], cdf)
            assert s < 0.05

    @pytest.mark.parametrize("regularized", [None, 10.0])
    def test_periodic(self, regularized):
        lower = -3.245
        upper = 5.123
        with self._model():
            dist = Periodic(
                "p",
                lower=lower,
                upper=upper,
                shape=(5, 2),
                regularized=regularized,
            )

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)
            assert np.all((lower <= samples) & (samples <= upper))

            logp = (
                Periodic.dist(lower=lower, upper=upper, shape=(5, 2))
                .logp(samples)
                .eval()
                .flatten()
            )
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        p = trace["p"]
        p = np.reshape(p, (len(p), -1))
        cdf = lambda x: np.clip((x - lower) / (upper - lower), 0, 1)  # NOQA
        for i in range(p.shape[1]):
            s, _ = kstest(p[:, i], cdf)
            assert s < 0.05

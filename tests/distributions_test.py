# -*- coding: utf-8 -*-

import logging
from collections import namedtuple

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
import pytest
from pymc3.tests.test_distributions import R, Unit, Vector
from pymc3.tests.test_transforms import check_transform, get_values
from scipy.stats import beta, halfnorm, kstest, rayleigh

from exoplanet.distributions import transforms as tr
from exoplanet.distributions.eccentricity import kipping13, vaneylen19
from exoplanet.distributions.physical import ImpactParameter, QuadLimbDark


class _Base:
    random_seed = 20160911

    def _sample(self, **kwargs):
        logger = logging.getLogger("pymc3")
        logger.propagate = False
        logger.setLevel(logging.ERROR)
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        return pm.sample(**kwargs)

    def _model(self, **kwargs):
        np.random.seed(self.random_seed)
        return pm.Model(**kwargs)


class TestEccentricity(_Base):
    random_seed = 19910626

    def test_kipping13(self):
        with self._model() as model:
            dist = kipping13("ecc", shape=(5, 2))
            assert "ecc_alpha" in model.named_vars
            assert "ecc_beta" in model.named_vars

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)

            assert np.all((0 <= samples) & (samples <= 1))

            trace = self._sample()

        ecc = trace["ecc"]
        assert np.all((0 <= ecc) & (ecc <= 1))

    def test_kipping13_all(self):
        with self._model():
            kipping13("ecc", fixed=True, shape=2)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        cdf = lambda x: beta.cdf(x, 1.12, 3.09)  # NOQA
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    def test_kipping13_long(self):
        with self._model():
            kipping13("ecc", fixed=True, long=True, shape=3)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        cdf = lambda x: beta.cdf(x, 1.12, 3.09)  # NOQA
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    def test_kipping13_short(self):
        with self._model():
            kipping13("ecc", fixed=True, long=False, shape=4)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        cdf = lambda x: beta.cdf(x, 0.697, 3.27)  # NOQA
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    @pytest.mark.parametrize(
        "kwargs",
        [dict(lower=0.1), dict(upper=0.5), dict(lower=0.3, upper=0.4)],
    )
    def test_kipping13_bounds(self, kwargs):
        with self._model():
            kipping13("ecc", **kwargs)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all(
            (kwargs.get("lower", 0.0) <= ecc)
            & (ecc <= kwargs.get("upper", 1.0))
        )

    @pytest.mark.parametrize("kwargs", [dict(), dict(multi=True)])
    def test_vaneylen19(self, kwargs):
        with self._model() as model:
            dist = vaneylen19("ecc", shape=(5, 2), **kwargs)

            if not kwargs.get("fixed", False):
                assert "ecc_sigma_gauss" in model.named_vars
                assert "ecc_sigma_rayleigh" in model.named_vars
                assert "ecc_frac" in model.named_vars

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 5, 2)
            assert np.all((0 <= samples) & (samples <= 1))

            trace = self._sample()

        ecc = trace["ecc"]
        assert np.all((0 <= ecc) & (ecc <= 1))

    def test_vaneylen19_single(self):
        with self._model():
            vaneylen19("ecc", fixed=True, multi=False, shape=2)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        f = 0.76
        cdf = lambda x: (  # NOQA
            (1 - f) * halfnorm.cdf(x, scale=0.049)
            + f * rayleigh.cdf(x, scale=0.26)
        )
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    def test_vaneylen19_multi(self):
        with self._model():
            vaneylen19("ecc", fixed=True, multi=True, shape=3)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        f = 0.08
        cdf = lambda x: (  # NOQA
            (1 - f) * halfnorm.cdf(x, scale=0.049)
            + f * rayleigh.cdf(x, scale=0.26)
        )
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    @pytest.mark.parametrize(
        "kwargs",
        [dict(lower=0.1), dict(upper=0.5), dict(lower=0.3, upper=0.4)],
    )
    def test_vaneylen19_bounds(self, kwargs):
        with self._model():
            vaneylen19("ecc", **kwargs)
            trace = self._sample()

        ecc = trace["ecc"].flatten()
        assert np.all(
            (kwargs.get("lower", 0.0) <= ecc)
            & (ecc <= kwargs.get("upper", 1.0))
        )


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


def test_quad_limb_dark_transform():
    values = get_values(
        tr.quad_limb_dark,
        Vector(R, 2),
        constructor=tt.vector,
        test=np.array([0.0, 0.0]),
    )
    domain = namedtuple("Domain", ["vals"])(values)
    check_transform(
        tr.quad_limb_dark,
        domain,
        constructor=tt.vector,
        test=np.array([0.0, 0.0]),
    )


def test_impact_parameter_transform():
    ror = np.float64(0.03)
    check_transform(
        tr.impact_parameter(ror),
        Unit * (1 + ror),
        test=0.5,
    )

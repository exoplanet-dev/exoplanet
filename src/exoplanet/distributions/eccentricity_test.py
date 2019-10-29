# -*- coding: utf-8 -*-

import numpy as np
import pytest
from scipy.stats import beta, halfnorm, kstest, rayleigh

from .base_test import _Base
from .eccentricity import kipping13, vaneylen19


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

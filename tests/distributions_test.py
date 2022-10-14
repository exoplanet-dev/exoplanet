import logging

import numpy as np
import pytest
from scipy.stats import beta, halfnorm, kstest, rayleigh

from exoplanet.compat import USING_PYMC3, pm
from exoplanet.distributions.distributions import (
    impact_parameter,
    quad_limb_dark,
)
from exoplanet.distributions.eccentricity import kipping13, vaneylen19


class _Base:
    random_seed = 20160911

    def _sample(self, **kwargs):
        logger = logging.getLogger("pymc3" if USING_PYMC3 else "pymc")
        logger.propagate = False
        logger.setLevel(logging.ERROR)
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        if USING_PYMC3:
            kwargs["return_inferencedata"] = True
            kwargs["compute_convergence_checks"] = False
        return pm.sample(**kwargs)

    def _model(self, **kwargs):
        np.random.seed(self.random_seed)
        return pm.Model(**kwargs)


class TestEccentricity(_Base):
    random_seed = 19910626

    def test_kipping13(self):
        with self._model() as model:
            dist = kipping13("ecc", shape=(5, 2))
            if USING_PYMC3:
                assert "ecc_alpha" in model.named_vars
                assert "ecc_beta" in model.named_vars
            else:
                assert "ecc::alpha" in model.named_vars
                assert "ecc::beta" in model.named_vars
            trace = self._sample()

        ecc = trace.posterior["ecc"].values
        assert np.all((0 <= ecc) & (ecc <= 1))

    def test_kipping13_all(self):
        with self._model():
            kipping13("ecc", fixed=True, shape=2)
            trace = self._sample()

        ecc = trace.posterior["ecc"].values.flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        cdf = lambda x: beta.cdf(x, 1.12, 3.09)  # NOQA
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    def test_kipping13_long(self):
        with self._model():
            kipping13("ecc", fixed=True, long=True, shape=3)
            trace = self._sample()

        ecc = trace.posterior["ecc"].values.flatten()
        assert np.all((0 <= ecc) & (ecc <= 1))

        cdf = lambda x: beta.cdf(x, 1.12, 3.09)  # NOQA
        s, p = kstest(ecc, cdf)
        assert s < 0.05

    def test_kipping13_short(self):
        with self._model():
            kipping13("ecc", fixed=True, long=False, shape=4)
            trace = self._sample()

        ecc = trace.posterior["ecc"].values.flatten()
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

        ecc = trace.posterior["ecc"].values.flatten()
        assert np.all(
            (kwargs.get("lower", 0.0) <= ecc)
            & (ecc <= kwargs.get("upper", 1.0))
        )

    @pytest.mark.parametrize("kwargs", [dict(), dict(multi=True)])
    def test_vaneylen19(self, kwargs):
        with self._model() as model:
            dist = vaneylen19("ecc", shape=(5, 2), **kwargs)

            if not kwargs.get("fixed", False):
                if USING_PYMC3:
                    assert "ecc_sigma_gauss" in model.named_vars
                    assert "ecc_sigma_rayleigh" in model.named_vars
                    assert "ecc_frac" in model.named_vars
                else:
                    assert "ecc::sigma_gauss" in model.named_vars
                    assert "ecc::sigma_rayleigh" in model.named_vars
                    assert "ecc::frac" in model.named_vars
            trace = self._sample()

        ecc = trace.posterior["ecc"].values
        assert np.all((0 <= ecc) & (ecc <= 1))

    def test_vaneylen19_single(self):
        with self._model():
            vaneylen19("ecc", fixed=True, multi=False, shape=2)
            trace = self._sample()

        ecc = trace.posterior["ecc"].values.flatten()
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

        ecc = trace.posterior["ecc"].values.flatten()
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

        ecc = trace.posterior["ecc"].values.flatten()
        assert np.all(
            (kwargs.get("lower", 0.0) <= ecc)
            & (ecc <= kwargs.get("upper", 1.0))
        )


class TestPhysical(_Base):
    random_seed = 19860925

    def test_quad_limb_dark(self):
        with self._model():
            quad_limb_dark("u")
            trace = self._sample()

        u1 = trace.posterior["u"].values[..., 0].flatten()
        u2 = trace.posterior["u"].values[..., 1].flatten()

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
        shape = (5, 2)
        with self._model():
            r = pm.Uniform("r", lower=lower, upper=upper, shape=shape)
            impact_parameter("b", r, shape=shape)
            trace = self._sample()

        u = trace.posterior["r"].values
        u = np.reshape(u, u.shape[:2] + (-1,))
        cdf = lambda x: np.clip((x - lower) / (upper - lower), 0, 1)  # NOQA
        for i in range(u.shape[-1]):
            s, p = kstest(u[..., i].flatten(), cdf)
            assert s < 0.05

        assert np.all(
            trace.posterior["b"].values <= 1 + trace.posterior["r"].values
        )

    @pytest.mark.skipif(
        USING_PYMC3, reason="Automatic shape inference doesn't work in PyMC3"
    )
    def test_impact_shape_inference(self):
        lower = 0.1
        upper = 1.0
        shape = (5, 2)
        with self._model():
            r = pm.Uniform("r", lower=lower, upper=upper, shape=shape)
            impact_parameter("b", r)
            trace = self._sample()

        assert trace.posterior["b"].values.shape[-2:] == shape

        u = trace.posterior["r"].values
        u = np.reshape(u, u.shape[:2] + (-1,))
        cdf = lambda x: np.clip((x - lower) / (upper - lower), 0, 1)  # NOQA
        for i in range(u.shape[-1]):
            s, p = kstest(u[..., i].flatten(), cdf)
            assert s < 0.05

        assert np.all(
            trace.posterior["b"].values <= 1 + trace.posterior["r"].values
        )

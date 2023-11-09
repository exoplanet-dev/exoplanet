import logging

import numpy as np
import pytest
from scipy.stats import beta, halfnorm, kstest, rayleigh

from exoplanet.compat import USING_PYMC3, pm
from exoplanet.distributions.distributions import (
    angle,
    impact_parameter,
    quad_limb_dark,
    unit_disk,
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
        kwargs["random_seed"] = kwargs.get("random_seed", self.random_seed)
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
            kipping13("ecc", fixed=False, shape=(5, 2))
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

    def test_kipping13_observed(self):
        with self._model() as model:
            secosw, sesinw = unit_disk(
                "secosw", "sesinw", shape=2, initval=0.01 * np.ones((2, 2))
            )
            ecc = pm.Deterministic("ecc", secosw**2 + sesinw**2)
            ecc_prior = kipping13("ecc_prior", shape=2, observed=ecc)

            # Is the prior added to the model as a potential?
            assert "ecc_prior" in model.named_vars
            assert ecc_prior in model.potentials

            # Is the prior taken into account when sampling the "posterior"?
            idata = self._sample()
            ecc = idata.posterior["ecc"].values.flatten()
            cdf = lambda x: beta.cdf(x, 1.12, 3.09)  # NOQA
            s, p = kstest(ecc, cdf)
            assert s < 0.05

    @pytest.mark.parametrize("kwargs", [dict(fixed=False), dict(multi=True)])
    def test_vaneylen19(self, kwargs):
        with self._model() as model:
            vaneylen19("ecc", shape=(5, 2), **kwargs)

            if not kwargs.get("fixed", True):
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

    def test_vaneylen19_observed(self):
        with self._model() as model:
            secosw, sesinw = unit_disk(
                "secosw", "sesinw", initval=0.01 * np.ones(2)
            )
            ecc = pm.Deterministic("ecc", secosw**2 + sesinw**2)
            ecc_prior = vaneylen19(
                "ecc_prior", fixed=True, multi=False, observed=ecc
            )

            # Is the prior added to the model as a potential?
            assert "ecc_prior" in model.named_vars
            assert ecc_prior in model.potentials

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


class TestUnitDisk(_Base):
    random_seed = 19930609

    def test_unit_disk(self):
        with self._model():
            h, k = unit_disk("h", "k")
            pm.Deterministic("ecc", h**2 + k**2)
            trace = self._sample()

        h_samples = trace.posterior["h"].values.flatten()
        k_samples = trace.posterior["k"].values.flatten()
        ecc_samples = trace.posterior["ecc"].values.flatten()

        # Check h and k in expected intervals
        assert np.all(h_samples < 1.0)
        assert np.all(h_samples > -1.0)
        assert np.all(k_samples < 1.0)
        assert np.all(k_samples > -1.0)

        # Check radius (eccentricity) is physical
        assert np.all(ecc_samples >= 0.0)
        assert np.all(ecc_samples < 1.0)

        # Check radius (eccentricity) is uniform
        cdf = lambda x: np.clip(x, 0, 1)  # NOQA
        s, p = kstest(ecc_samples, cdf)
        assert s < 0.05

    @pytest.mark.parametrize(
        "shape",
        [2, (1, 3), (4, 5)],
    )
    def test_unit_disk_shape(self, shape):
        shape_as_tuple = (shape,) if isinstance(shape, int) else shape
        with self._model():
            h, k = unit_disk("h", "k", shape=shape)

        assert h.type.shape == shape_as_tuple
        assert k.type.shape == h.type.shape

    @pytest.mark.parametrize(
        "shape",
        [3, (4, 1), (4, 5)],
    )
    def test_unit_disk_initval(self, shape):
        shape_as_tuple = (shape,) if isinstance(shape, int) else shape

        with self._model():
            h, k = unit_disk(
                "h", "k", shape=shape, initval=np.zeros((2,) + shape_as_tuple)
            )

        assert h.type.shape == shape_as_tuple
        assert k.type.shape == h.type.shape


class TestAngle(_Base):
    random_seed = 19900101

    def test_angle(self):
        with self._model():
            angle("theta")
            trace = self._sample()

        theta_samples = trace.posterior["theta"].values.flatten()

        # Check h and k in expected intervals
        assert np.all(np.abs(theta_samples) < np.pi)

        # Check theta is uniform
        cdf = lambda x: np.clip(  # NOQA
            (x + np.pi) / (2 * np.pi), -np.pi, np.pi
        )
        s, p = kstest(theta_samples, cdf)
        assert s < 0.05

    @pytest.mark.parametrize(
        "shape",
        [2, (1, 3), (4, 5)],
    )
    def test_angle_shape(self, shape):
        shape_as_tuple = (shape,) if isinstance(shape, int) else shape
        with self._model():
            theta = angle("theta", shape=shape)

        assert theta.type.shape == shape_as_tuple

    @pytest.mark.parametrize(
        "shape",
        [3, (4, 1), (4, 5)],
    )
    def test_angle_initval(self, shape):
        shape_as_tuple = (shape,) if isinstance(shape, int) else shape
        with self._model():
            theta = angle("theta", shape=shape, initval=np.zeros(shape))

        assert theta.type.shape == shape_as_tuple


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

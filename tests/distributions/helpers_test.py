# -*- coding: utf-8 -*-

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
import pytest
from aesara_theano_fallback import aesara as theano
from scipy.stats import invgamma

from exoplanet.distributions.helpers import (
    estimate_inverse_gamma_parameters,
    get_log_abs_det_jacobian,
)
from exoplanet.orbits import KeplerianOrbit


def test_get_log_abs_det_jacobian():
    # Sorry this one's a bit convoluted...
    np.random.seed(20200409)

    log_duration = tt.dscalar()
    log_duration.tag.test_value = 0.1
    r_star = tt.dscalar()
    r_star.tag.test_value = 0.73452
    orbit = KeplerianOrbit(
        period=10.0,
        t0=0.0,
        b=0.5,
        duration=tt.exp(log_duration),
        r_star=r_star,
    )
    log_m = tt.log(orbit.m_star)
    log_rho = tt.log(orbit.rho_star)
    log_abs_det = get_log_abs_det_jacobian(
        [log_duration, r_star], [log_m, log_rho]
    )

    func = theano.function(
        [log_duration, r_star], tt.stack((log_m, log_rho, log_abs_det))
    )
    in_args = [log_duration.tag.test_value, r_star.tag.test_value]
    grad = []
    for n in range(2):
        grad.append(
            np.append(
                *theano.gradient.numeric_grad(
                    lambda *args: func(*args)[n], in_args
                ).gf
            )
        )

    assert np.allclose(np.linalg.slogdet(grad)[1], func(*in_args)[-1])


@pytest.mark.parametrize(
    "lower, upper, target",
    [(1.0, 2.0, 0.01), (0.01, 0.1, 0.1), (10.0, 25.0, 0.01)],
)
def test_estimate_inverse_gamma_parameters(lower, upper, target):
    np.random.seed(20200409)

    params = estimate_inverse_gamma_parameters(lower, upper, target=target)
    dist = invgamma(params["alpha"], scale=params["beta"])
    assert np.allclose(dist.cdf(lower), target)
    assert np.allclose(1 - dist.cdf(upper), target)

    samples = pm.InverseGamma.dist(**params).random(size=10000)
    assert np.allclose(
        (samples < lower).sum() / len(samples), target, atol=1e-2
    )
    assert np.allclose(
        (samples > upper).sum() / len(samples), target, atol=1e-2
    )

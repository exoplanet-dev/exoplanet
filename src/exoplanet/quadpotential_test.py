# -*- coding: utf-8 -*-

import numpy as np

from .quadpotential import QuadPotentialDenseAdapt, _WeightedCovariance


def get_cov(ndim):
    L = np.random.randn(ndim, ndim)
    L[np.triu_indices_from(L, 1)] = 0.0
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    return np.dot(L, L.T)


def test_weighted_covariance(ndim=10, seed=5432):
    np.random.seed(seed)

    cov = get_cov(ndim)
    mean = np.random.randn(ndim)

    samples = np.random.multivariate_normal(mean, cov, size=100)
    mu_est0 = np.mean(samples, axis=0)
    cov_est0 = np.cov(samples, rowvar=0)

    est = _WeightedCovariance(ndim)
    for sample in samples:
        est.add_sample(sample, 1)
    mu_est = est.current_mean()
    cov_est = est.current_covariance()

    assert np.allclose(mu_est, mu_est0)
    assert np.allclose(cov_est, cov_est0)

    # Make sure that the weighted estimate also works
    est2 = _WeightedCovariance(
        ndim,
        np.mean(samples[:10], axis=0),
        np.cov(samples[:10], rowvar=0, bias=True),
        10,
    )
    for sample in samples[10:]:
        est2.add_sample(sample, 1)
    mu_est2 = est2.current_mean()
    cov_est2 = est2.current_covariance()

    assert np.allclose(mu_est2, mu_est0)
    assert np.allclose(cov_est2, cov_est0)


def test_draw_samples(ndim=10, seed=8976):
    # NB: the covariance of the generated samples should be the *inverse* of
    # the given covariance because these are momentum samples!
    np.random.seed(seed)
    cov = get_cov(ndim)
    cov[np.diag_indices_from(cov)] += 0.1
    invcov = np.linalg.inv(cov)

    np.random.seed(seed)
    sample_cov0 = np.cov(
        np.random.multivariate_normal(np.zeros(ndim), invcov, size=10000),
        rowvar=0,
    )

    np.random.seed(seed)
    pot = QuadPotentialDenseAdapt(ndim, np.zeros(ndim), cov, 1)
    samples = [pot.random() for n in range(10000)]
    sample_cov = np.cov(samples, rowvar=0)
    assert np.all(np.abs(sample_cov - sample_cov0) < 0.1)


def test_sample_p(seed=4566):
    # ref: https://github.com/stan-dev/stan/pull/2672
    np.random.seed(seed)
    m = np.array([[3.0, -2.0], [-2.0, 4.0]])
    m_inv = np.linalg.inv(m)

    var = np.array(
        [
            [2 * m[0, 0], m[1, 0] * m[1, 0] + m[1, 1] * m[0, 0]],
            [m[0, 1] * m[0, 1] + m[1, 1] * m[0, 0], 2 * m[1, 1]],
        ]
    )

    n_samples = 1000
    pot = QuadPotentialDenseAdapt(2, np.zeros(2), m_inv, 1)
    samples = [pot.random() for n in range(n_samples)]
    sample_cov = np.cov(samples, rowvar=0)

    # Covariance matrix within 5 sigma of expected value
    # (comes from a Wishart distribution)
    assert np.all(np.abs(m - sample_cov) < 5 * np.sqrt(var / n_samples))

# -*- coding: utf-8 -*-

import numpy as np
import pytest
import theano
import theano.tensor as tt
from scipy.linalg import cho_solve, cholesky

from . import terms
from .celerite import GP


def test_broadcast_dim():
    logS0 = tt.scalar()
    logw0 = tt.scalar()
    logQ = tt.scalar()
    logS0.tag.test_value = -5.0
    logw0.tag.test_value = -2.0
    logQ.tag.test_value = 1.0
    kernel = terms.SHOTerm(S0=tt.exp(logS0), w0=tt.exp(logw0), Q=tt.exp(logQ))

    x = tt.vector()
    y = tt.vector()
    diag = tt.vector()
    x.tag.test_value = np.zeros(2)
    y.tag.test_value = np.zeros(2)
    diag.tag.test_value = np.ones(2)
    gp = GP(kernel, x, diag, J=2)
    loglike = gp.log_likelihood(y)

    args = [logS0, logw0, logQ, x, y, diag]
    grad = theano.function(args, theano.grad(loglike, args))

    np.random.seed(42)
    N = 50
    x = np.sort(10 * np.random.rand(N))
    y = np.sin(x)
    diag = np.random.rand(N)
    grad(-5.0, -2.0, 1.0, x, y, diag)


def test_drop_non_broadcastable():
    np.random.seed(123)
    mean = tt.dscalar()
    mean.tag.test_value = 0.1
    gp = GP(terms.RealTerm(a=1.0, c=1.0), np.linspace(0, 10, 50), np.ones(50))
    arg = np.random.rand(50) - mean
    res = gp.apply_inverse(arg[:, None])
    theano.grad(tt.sum(res), [arg])
    theano.grad(tt.sum(arg), [mean])
    theano.grad(tt.sum(res), [mean])


def _get_theano_kernel(celerite_kernel):
    import celerite.terms as cterms

    if isinstance(celerite_kernel, cterms.TermSum):
        result = _get_theano_kernel(celerite_kernel.terms[0])
        for k in celerite_kernel.terms[1:]:
            result += _get_theano_kernel(k)
        return result
    elif isinstance(celerite_kernel, cterms.TermProduct):
        return _get_theano_kernel(celerite_kernel.k1) * _get_theano_kernel(
            celerite_kernel.k2
        )
    elif isinstance(celerite_kernel, cterms.RealTerm):
        return terms.RealTerm(
            log_a=celerite_kernel.log_a, log_c=celerite_kernel.log_c
        )
    elif isinstance(celerite_kernel, cterms.ComplexTerm):
        if not celerite_kernel.fit_b:
            return terms.ComplexTerm(
                log_a=celerite_kernel.log_a,
                b=0.0,
                log_c=celerite_kernel.log_c,
                log_d=celerite_kernel.log_d,
            )
        return terms.ComplexTerm(
            log_a=celerite_kernel.log_a,
            log_b=celerite_kernel.log_b,
            log_c=celerite_kernel.log_c,
            log_d=celerite_kernel.log_d,
        )
    elif isinstance(celerite_kernel, cterms.SHOTerm):
        return terms.SHOTerm(
            log_S0=celerite_kernel.log_S0,
            log_Q=celerite_kernel.log_Q,
            log_w0=celerite_kernel.log_omega0,
        )
    elif isinstance(celerite_kernel, cterms.Matern32Term):
        return terms.Matern32Term(
            log_sigma=celerite_kernel.log_sigma,
            log_rho=celerite_kernel.log_rho,
        )
    raise NotImplementedError()


@pytest.mark.parametrize(
    "celerite_kernel",
    [
        "cterms.RealTerm(log_a=0.1, log_c=0.5) + "
        "cterms.RealTerm(log_a=-0.1, log_c=0.7)",
        "cterms.ComplexTerm(log_a=0.1, log_c=0.5, log_d=0.1)",
        "cterms.ComplexTerm(log_a=0.1, log_b=-0.2, log_c=0.5, log_d=0.1)",
        "cterms.SHOTerm(log_S0=0.1, log_Q=-1, log_omega0=0.5)",
        "cterms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5)",
        "cterms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) + "
        "cterms.RealTerm(log_a=0.1, log_c=0.4)",
        "cterms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) * "
        "cterms.RealTerm(log_a=0.1, log_c=0.4)",
        "cterms.Matern32Term(log_sigma=0.1, log_rho=0.4)",
    ],
)
def test_gp(celerite_kernel, seed=1234):
    import celerite
    import celerite.terms as cterms  # NOQA

    celerite_kernel = eval(celerite_kernel)
    np.random.seed(seed)
    x = np.sort(np.random.rand(100))
    t = np.sort(np.random.rand(50))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)
    diag = yerr ** 2

    celerite_gp = celerite.GP(celerite_kernel)
    celerite_gp.compute(x, yerr)
    celerite_loglike = celerite_gp.log_likelihood(y)
    celerite_mu, celerite_cov = celerite_gp.predict(y)
    _, celerite_var = celerite_gp.predict(y, return_cov=False, return_var=True)

    celerite_mu_t, celerite_cov_t = celerite_gp.predict(y, t=t)
    _, celerite_var_t = celerite_gp.predict(
        y, t=t, return_cov=False, return_var=True
    )

    kernel = _get_theano_kernel(celerite_kernel)
    gp = GP(kernel, x, diag)
    loglike = gp.log_likelihood(y).eval()

    assert np.allclose(loglike, celerite_loglike)

    mu = gp.predict()
    _, var = gp.predict(return_var=True)
    _, cov = gp.predict(return_cov=True)
    assert np.allclose(mu.eval(), celerite_mu)
    assert np.allclose(var.eval(), celerite_var)
    assert np.allclose(cov.eval(), celerite_cov)

    mu = gp.predict(t)
    _, var = gp.predict(t, return_var=True)
    _, cov = gp.predict(t, return_cov=True)
    assert np.allclose(mu.eval(), celerite_mu_t)
    assert np.allclose(var.eval(), celerite_var_t)
    assert np.allclose(cov.eval(), celerite_cov_t)


def test_integrated_diag(seed=1234):
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 100, 100))
    dt = 0.4 * np.min(np.diff(x))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    diag = yerr ** 2

    kernel = terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_w0=0.5)
    kernel += terms.RealTerm(log_a=0.1, log_c=0.4)

    a = kernel.get_celerite_matrices(x, diag)[0].eval()
    k0 = kernel.value(tt.zeros(1)).eval()
    assert np.allclose(a, k0 + diag)

    kernel = terms.IntegratedTerm(kernel, dt)
    a = kernel.get_celerite_matrices(x, diag)[0].eval()
    k0 = kernel.value(tt.zeros(1)).eval()
    assert np.allclose(a, k0 + diag)


def _check_model(kernel, x, diag, y):
    gp = GP(kernel, x, diag)
    loglike = gp.log_likelihood(y).eval()
    Ly = gp.dot_l(y[:, None]).eval()

    K = kernel.value(x[:, None] - x[None, :]).eval()
    K[np.diag_indices_from(K)] += diag
    factor = (cholesky(K, overwrite_a=True, lower=True), True)

    assert np.allclose(np.dot(factor[0], y[:, None]), Ly)

    loglike0 = -np.sum(np.log(np.diag(factor[0])))
    loglike0 -= 0.5 * len(x) * np.log(2 * np.pi)
    loglike0 -= 0.5 * np.dot(y, cho_solve(factor, y))

    assert np.allclose(loglike, loglike0)


@pytest.mark.parametrize(
    "kernel",
    [
        terms.RealTerm(log_a=0.1, log_c=0.5),
        terms.RealTerm(log_a=0.1, log_c=0.5)
        + terms.RealTerm(log_a=-0.1, log_c=0.7),
        terms.ComplexTerm(log_a=0.1, b=0.0, log_c=0.5, log_d=0.1),
        terms.ComplexTerm(log_a=0.1, log_b=-0.2, log_c=0.5, log_d=0.1),
        terms.SHOTerm(log_S0=0.1, log_Q=-1, log_w0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_w0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_w0=0.5)
        + terms.RealTerm(log_a=0.1, log_c=0.4),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_w0=0.5)
        * terms.RealTerm(log_a=0.1, log_c=0.4),
        terms.Matern32Term(log_sigma=0.1, log_rho=0.4),
    ],
)
def test_integrated(kernel, seed=1234):
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 100, 100))
    dt = 0.4 * np.min(np.diff(x))
    y = np.sin(x)
    yerr = np.random.uniform(0.1, 0.5, len(x))
    diag = yerr ** 2

    _check_model(kernel, x, diag, y)

    kernel = terms.IntegratedTerm(kernel, dt)
    _check_model(kernel, x, diag, y)


def test_sho_reparam(seed=6083):
    S0 = 10.0
    w0 = 0.5
    Q = 3.2
    kernel1 = terms.SHOTerm(S0=S0, w0=w0, Q=Q)
    kernel2 = terms.SHOTerm(Sw4=S0 * w0 ** 4, w0=w0, Q=Q)
    func1 = theano.function([], kernel1.coefficients)
    func2 = theano.function([], kernel2.coefficients)
    for a, b in zip(func1(), func2()):
        assert np.allclose(a, b)

    kernel2 = terms.SHOTerm(log_Sw4=np.log(S0) + 4 * np.log(w0), w0=w0, Q=Q)
    func2 = theano.function([], kernel2.coefficients)
    for a, b in zip(func1(), func2()):
        assert np.allclose(a, b)

    Q = 0.1
    kernel1 = terms.SHOTerm(S0=S0, w0=w0, Q=Q)
    kernel2 = terms.SHOTerm(Sw4=S0 * w0 ** 4, w0=w0, Q=Q)
    func1 = theano.function([], kernel1.coefficients)
    func2 = theano.function([], kernel2.coefficients)
    for a, b in zip(func1(), func2()):
        assert np.allclose(a, b)

    kernel2 = terms.SHOTerm(log_Sw4=np.log(S0) + 4 * np.log(w0), w0=w0, Q=Q)
    func2 = theano.function([], kernel2.coefficients)
    for a, b in zip(func1(), func2()):
        assert np.allclose(a, b)


def test_fortran_order(seed=5091986):
    np.random.seed(seed)

    kernel = terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_w0=0.5)

    x = np.sort(np.random.uniform(0, 100, 100))
    y = np.sin(x)
    yerr = np.random.uniform(0.1, 0.5, len(x))
    diag = yerr ** 2

    gp = GP(kernel, x, diag)
    loglike = gp.log_likelihood(y).eval()
    loglike_f = gp.log_likelihood(np.asfortranarray(y)).eval()
    assert np.allclose(loglike, loglike_f)

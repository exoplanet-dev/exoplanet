# -*- coding: utf-8 -*-

__all__ = ["QuadPotentialDenseAdapt", "get_dense_nuts_step", "sample"]

import numpy as np
import pymc3 as pm
import theano
from pymc3.model import all_continuous, modelcontext
from pymc3.step_methods.hmc.quadpotential import QuadPotential
from pymc3.step_methods.step_sizes import DualAverageAdaptation
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from .utils import logger


class QuadPotentialDenseAdapt(QuadPotential):
    """Adapt a dense mass matrix from the sample covariances."""

    def __init__(
        self,
        n,
        initial_mean,
        initial_cov=None,
        initial_weight=0,
        adaptation_window=101,
        doubling=True,
        dtype=None,
    ):
        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_cov is not None and initial_cov.shape != (n, n):
            raise ValueError(
                "Wrong shape for initial_cov: expected %s got %s"
                % (n, initial_cov.shape)
            )
        if len(initial_mean) != n:
            raise ValueError(
                "Wrong shape for initial_mean: expected %s got %s"
                % (n, len(initial_mean))
            )

        if dtype is None:
            dtype = theano.config.floatX

        if initial_cov is None:
            initial_cov = np.eye(n, dtype=dtype)
            initial_weight = 1

        self.dtype = dtype
        self._n = n
        self._cov = np.array(initial_cov, dtype=self.dtype, copy=True)
        self._cov_theano = theano.shared(self._cov)
        self._chol = cholesky(self._cov, lower=True)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, initial_mean, initial_cov, initial_weight, self.dtype
        )
        self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

        self._doubling = doubling
        self._adaptation_window = int(adaptation_window)
        self._previous_update = 0

    def velocity(self, x, out=None):
        return np.dot(self._cov, x, out=out)

    def energy(self, x, velocity=None):
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * np.dot(x, velocity)

    def velocity_energy(self, x, v_out):
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    def random(self):
        vals = np.random.normal(size=self._n).astype(self.dtype)
        return solve_triangular(self._chol.T, vals, overwrite_b=True)

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        try:
            self._chol = cholesky(self._cov, lower=True)
        except LinAlgError as error:
            self._chol_error = error
        self._cov_theano.set_value(self._cov)

    def update(self, sample, grad, tune):
        if not tune:
            return

        self._foreground_cov.add_sample(sample, weight=1)
        self._background_cov.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_cov)

        # Steps since previous update
        delta = self._n_samples - self._previous_update
        if delta >= self._adaptation_window:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(
                self._n, dtype=self.dtype
            )

            self._previous_update = self._n_samples
            if self._doubling:
                self._adaptation_window *= 2

        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError("{0}".format(self._chol_error))


class _WeightedCovariance:
    """Online algorithm for computing mean and covariance."""

    def __init__(
        self,
        nelem,
        initial_mean=None,
        initial_covariance=None,
        initial_weight=0,
        dtype="d",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype="d")
        else:
            self.mean = np.array(initial_mean, dtype="d", copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem, dtype="d")
        else:
            self.raw_cov = np.array(initial_covariance, dtype="d", copy=True)

        self.raw_cov[:] *= self.n_samples

        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_cov[:] += weight * new_diff[:, None] * old_diff[None, :]

    def current_covariance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        else:
            return (self.raw_cov / (self.n_samples - 1)).astype(self._dtype)

    def current_mean(self):
        return np.array(self.mean, dtype=self._dtype)


class WindowedQuadPotentialDenseAdapt(QuadPotentialDenseAdapt):
    """Adapt a dense mass matrix from the windowed sample covariances"""

    def __init__(self, n, start_step, update_steps, dtype=None):
        if dtype is None:
            dtype = theano.config.floatX

        initial_cov = np.eye(n, dtype=dtype)

        self.start_step = int(start_step)
        self.update_steps = np.atleast_1d(update_steps).astype(int)

        self.dtype = dtype
        self._n = n
        self._cov = np.array(initial_cov, dtype=self.dtype, copy=True)
        self._cov_theano = theano.shared(self._cov)
        self._chol = cholesky(self._cov, lower=True)
        self._chol_error = None

        self._cov_est = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)

        # Regularize
        n = weightvar.n_samples
        self._cov *= n / (n + 5.0)
        self._cov[np.diag_indices_from(self._cov)] += 1e-3 * 5.0 / (n + 5.0)

        try:
            self._chol = cholesky(self._cov, lower=True)
        except LinAlgError as error:
            self._chol_error = error
        self._cov_theano.set_value(self._cov)

    def update(self, sample, grad, tune):
        if not tune:
            return

        # Don't start tracking samples until after the warmup
        if self._n_samples < self.start_step:
            self._n_samples += 1
            return

        # Keep track of this sample
        self._cov_est.add_sample(sample, weight=1)

        # Update the covariance at the correct steps
        if self._n_samples in self.update_steps:
            self._update_from_weightvar(self._cov_est)
            self._cov_est = _WeightedCovariance(self._n, dtype=self.dtype)

        self._n_samples += 1


class WindowedDualAverageAdaptation(DualAverageAdaptation):
    def __init__(self, update_steps, initial_step, *args, **kwargs):
        self.update_steps = np.atleast_1d(update_steps).astype(int)
        super(WindowedDualAverageAdaptation, self).__init__(
            initial_step, *args, **kwargs
        )
        self._initial_step = initial_step
        self._n_samples = 0

    def reset(self):
        self._hbar = 0.0
        self._log_step = np.log(self._initial_step)
        self._log_bar = self._log_step
        self._count = 1

    def update(self, accept_stat, tune):
        if self._n_samples in self.update_steps:
            self._n_samples += 1
            self.reset()
            return

        self._n_samples += 1
        super(WindowedDualAverageAdaptation, self).update(accept_stat, tune)


def build_schedule(
    tune, warmup_window=50, adapt_window=50, cooldown_window=50
):
    if warmup_window + adapt_window + cooldown_window > tune:
        logger.warn(
            "there are not enough tuning steps to accomodate the tuning "
            "schedule; assigning automatically as 20%/70%/10%"
        )
        warmup_window = np.ceil(0.2 * tune).astype(int)
        cooldown_window = np.ceil(0.1 * tune).astype(int)
        adapt_window = tune - warmup_window - cooldown_window

    t = warmup_window
    delta = adapt_window
    update_steps = []
    while t < tune - cooldown_window:
        t += delta
        delta = 2 * delta
        if t + delta > tune - cooldown_window:
            update_steps.append(tune - cooldown_window)
            break
        update_steps.append(t)

    update_steps = np.array(update_steps, dtype=int)
    if np.any(update_steps) <= 0:
        raise ValueError("invalid tuning schedule")

    return warmup_window, update_steps


def sample(
    draws=1000,
    tune=1000,
    model=None,
    warmup_window=50,
    adapt_window=50,
    cooldown_window=50,
    target_accept=0.9,
    gamma=0.05,
    k=0.75,
    t0=10,
    step_kwargs=None,
    **kwargs,
):
    model = modelcontext(model)

    if not all_continuous(model.vars):
        raise ValueError(
            "NUTS can only be used for models with only "
            "continuous variables."
        )

    warmup_window, update_steps = build_schedule(
        tune,
        warmup_window=warmup_window,
        adapt_window=adapt_window,
        cooldown_window=cooldown_window,
    )

    potential = WindowedQuadPotentialDenseAdapt(
        model.ndim, warmup_window, update_steps
    )

    if step_kwargs is None:
        step_kwargs = {}
    step = pm.NUTS(
        potential=potential,
        model=model,
        target_accept=target_accept,
        **step_kwargs,
    )
    step.step_adapt = WindowedDualAverageAdaptation(
        np.append(warmup_window, update_steps),
        step.step_size,
        target_accept,
        gamma,
        k,
        t0,
    )

    kwargs["step"] = step
    return pm.sample(draws=draws, tune=tune, model=model, **kwargs)


def get_dense_nuts_step(
    start=None,
    adaptation_window=101,
    doubling=True,
    initial_weight=10,
    use_hessian=False,
    use_hessian_diag=False,
    hessian_regularization=1e-8,
    model=None,
    **kwargs,
):
    """Get a NUTS step function with a dense mass matrix

    The entries in the mass matrix will be tuned based on the sample
    covariances during tuning. All extra arguments are passed directly to
    ``pymc3.NUTS``.

    Args:
        start (dict, optional): A starting point in parameter space. If not
            provided, the model's ``test_point`` is used.
        adaptation_window (int, optional): The (initial) size of the window
            used for sample covariance estimation.
        doubling (bool, optional): If ``True`` (default) the adaptation window
            is doubled each time the matrix is updated.

    """
    model = modelcontext(model)

    if not all_continuous(model.vars):
        raise ValueError(
            "NUTS can only be used for models with only "
            "continuous variables."
        )

    if start is None:
        start = model.test_point
    mean = model.dict_to_array(start)

    if use_hessian or use_hessian_diag:
        try:
            import numdifftools as nd
        except ImportError:
            raise ImportError(
                "The 'numdifftools' package is required for Hessian "
                "computations"
            )

        logger.info("Numerically estimating Hessian matrix")
        if use_hessian_diag:
            hess = nd.Hessdiag(model.logp_array)(mean)
            var = np.diag(-1.0 / hess)
        else:
            hess = nd.Hessian(model.logp_array)(mean)
            var = -np.linalg.inv(hess)

        factor = 1
        success = False
        while not success:
            var[np.diag_indices_from(var)] += factor * hessian_regularization

            try:
                np.linalg.cholesky(var)
            except np.linalg.LinAlgError:
                factor *= 2
            else:
                success = True

    else:
        var = np.eye(len(mean))

    potential = QuadPotentialDenseAdapt(
        model.ndim,
        mean,
        var,
        initial_weight,
        adaptation_window=adaptation_window,
        doubling=doubling,
    )

    return pm.NUTS(potential=potential, model=model, **kwargs)

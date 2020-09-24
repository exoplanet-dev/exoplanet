# -*- coding: utf-8 -*-

__all__ = ["QuadPotentialDenseAdapt", "get_dense_nuts_step", "sample"]

import numpy as np
import pymc3 as pm
import theano
from pymc3.model import all_continuous, modelcontext
from pymc3.step_methods.hmc.quadpotential import QuadPotential
from pymc3.step_methods.step_sizes import DualAverageAdaptation
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from .utils import deprecated, logger


class QuadPotentialDenseAdapt(QuadPotential):
    """Adapt a dense mass matrix from the sample covariances."""

    def __init__(
        self,
        n,
        initial_mean=None,
        initial_cov=None,
        initial_weight=0,
        adaptation_window=101,
        doubling=True,
        update_steps=None,
        dtype="float64",
    ):
        if initial_mean is None:
            initial_mean = np.zeros(n, dtype=dtype)
        if initial_cov is None:
            initial_cov = np.eye(n, dtype=dtype)
            initial_weight = 1

        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean is not None and initial_mean.ndim != 1:
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

        # For backwards compatibility
        self._doubling = doubling
        self._adaptation_window = int(adaptation_window)
        self._previous_update = 0

        # New interface
        if update_steps is None:
            self._update_steps = None
        else:
            self._update_steps = np.atleast_1d(update_steps).astype(int)

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
        except (LinAlgError, ValueError) as error:
            self._chol_error = error
        self._cov_theano.set_value(self._cov)

    def update(self, sample, grad, tune):
        if not tune:
            return

        self._foreground_cov.add_sample(sample, weight=1)
        self._background_cov.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_cov)

        # Support the two methods for updating the mass matrix
        delta = self._n_samples - self._previous_update
        do_update = (
            self._update_steps is not None
            and self._n_samples in self._update_steps
        ) or (self._update_steps is None and delta >= self._adaptation_window)
        if do_update:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(
                self._n, dtype=self.dtype
            )

            if self._update_steps is None:
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
        dtype="float64",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype=dtype)
        else:
            self.mean = np.array(initial_mean, dtype=dtype, copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem, dtype=dtype)
        else:
            self.raw_cov = np.array(initial_covariance, dtype=dtype, copy=True)

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


class WindowedDualAverageAdaptation(DualAverageAdaptation):
    def __init__(self, update_steps, initial_step, target, *args, **kwargs):
        self.update_steps = np.atleast_1d(update_steps).astype(int)
        self.targets = np.atleast_1d(target) + np.zeros_like(self.update_steps)
        super(WindowedDualAverageAdaptation, self).__init__(
            initial_step, self.targets[0], *args, **kwargs
        )
        self._initial_step = initial_step
        self._n_samples = 0

    def reset(self):
        self._hbar = 0.0
        self._log_step = np.log(self._initial_step)
        self._log_bar = self._log_step
        self._mu = np.log(10 * self._initial_step)
        self._count = 1
        self._tuned_stats = []

    def update(self, accept_stat, tune):
        if self._n_samples in self.update_steps:
            self._target = float(
                self.targets[np.where(self.update_steps == self._n_samples)]
            )
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

    return np.append(warmup_window, update_steps)


@deprecated("the sample function from the pymc3-ext library")
def sample(
    *,
    draws=1000,
    tune=1000,
    model=None,
    step_kwargs=None,
    warmup_window=50,
    adapt_window=50,
    cooldown_window=100,
    initial_accept=None,
    target_accept=0.9,
    gamma=0.05,
    k=0.75,
    t0=10,
    **kwargs,
):
    # Check that we're in a model context and that all the variables are
    # continuous
    model = modelcontext(model)
    if not all_continuous(model.vars):
        raise ValueError(
            "NUTS can only be used for models with only "
            "continuous variables."
        )
    start = kwargs.get("start", None)
    if start is None:
        start = model.test_point
    mean = model.dict_to_array(start)

    update_steps = build_schedule(
        tune,
        warmup_window=warmup_window,
        adapt_window=adapt_window,
        cooldown_window=cooldown_window,
    )

    potential = QuadPotentialDenseAdapt(
        model.ndim,
        initial_mean=mean,
        initial_weight=10,
        update_steps=update_steps,
    )

    if "step" in kwargs:
        step = kwargs["step"]
    else:
        if step_kwargs is None:
            step_kwargs = {}
        step = pm.NUTS(
            potential=potential,
            model=model,
            target_accept=target_accept,
            **step_kwargs,
        )

    if "target_accept" in step_kwargs and target_accept is not None:
        raise ValueError(
            "'target_accept' cannot be given as a keyword argument and in "
            "'step_kwargs'"
        )
    target_accept = step_kwargs.pop("target_accept", target_accept)
    if initial_accept is None:
        target = target_accept
    else:
        if initial_accept > target_accept:
            raise ValueError(
                "initial_accept must be less than or equal to target_accept"
            )
        target = initial_accept + (target_accept - initial_accept) * np.sqrt(
            np.arange(len(update_steps)) / (len(update_steps) - 1)
        )
    step.step_adapt = WindowedDualAverageAdaptation(
        update_steps, step.step_size, target, gamma, k, t0
    )

    kwargs["step"] = step
    return pm.sample(draws=draws, tune=tune, model=model, **kwargs)


@deprecated("the init='full_adapt' argument to pm.sample")
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
        initial_mean=mean,
        initial_cov=var,
        initial_weight=initial_weight,
        adaptation_window=adaptation_window,
        doubling=doubling,
    )

    return pm.NUTS(potential=potential, model=model, **kwargs)

# -*- coding: utf-8 -*-

__all__ = ["QuadPotentialDenseAdapt", "get_dense_nuts_step"]

import numpy as np
import pymc3 as pm
import theano
from pymc3.model import all_continuous, modelcontext
from pymc3.step_methods.hmc.quadpotential import QuadPotential
from scipy.linalg import LinAlgError, cholesky, solve_triangular


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


def get_dense_nuts_step(
    start=None, adaptation_window=101, doubling=True, model=None, **kwargs
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
    var = np.eye(len(mean))
    potential = QuadPotentialDenseAdapt(
        model.ndim,
        mean,
        var,
        10,
        adaptation_window=adaptation_window,
        doubling=doubling,
    )

    return pm.NUTS(potential=potential, model=model, **kwargs)

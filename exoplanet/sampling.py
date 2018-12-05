# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["PyMC3Sampler"]

import logging

import numpy as np

import pymc3 as pm
from pymc3.step_methods.hmc import quadpotential as quad


class PyMC3Sampler(object):
    """A sampling wrapper for PyMC3 with support for dense mass matrices

    This schedule is based on the method used by as described in Section 34.2
    of the `Stan Manual <http://mc-stan.org/users/documentation/>`_.

    Args:
        start (int): The number of steps to run as an initial burn-in to find
            the typical set.
        window (int): The length of the first mass matrix tuning phase.
            Subsequent tuning windows will be powers of two times this size.
        finish (int): The number of tuning steps to run to learn the step size
            after tuning the mass matrix.
        dense (bool): Fit for the off-diagonal elements of the mass matrix.

    """

    # Ref: src/stan/mcmc/windowed_adaptation.hpp in stan repo

    def __init__(self, start=75, finish=50, window=25, dense=True):
        self.dense = dense

        self.start = int(start)
        self.finish = int(finish)
        self.window = int(window)
        self.count = 0
        self._current_step = None
        self._current_trace = None

    def get_step_for_trace(self, trace=None, model=None,
                           regular_window=0, regular_variance=1e-3,
                           **kwargs):
        """Get a PyMC3 NUTS step tuned for a given burn-in trace

        Args:
            trace: The ``MultiTrace`` output from a previous run of
                ``pymc3.sample``.
            regular_window: The weight (in units of number of steps) to use
                when regularizing the mass matrix estimate.
            regular_variance: The amplitude of the regularization for the mass
                matrix. This will be added to the diagonal of the covariance
                matrix with weight given by ``regular_window``.

        """
        model = pm.modelcontext(model)

        # If not given, use the trivial metric
        if trace is None or model.ndim == 1:
            potential = quad.QuadPotentialDiag(np.ones(model.ndim))

        else:
            # Loop over samples and convert to the relevant parameter space;
            # I'm sure that there's an easier way to do this, but I don't know
            # how to make something work in general...
            N = len(trace) * trace.nchains
            samples = np.empty((N, model.ndim))
            i = 0
            for chain in trace._straces.values():
                for p in chain:
                    samples[i] = model.bijection.map(p)
                    i += 1

            if self.dense:
                # Compute the regularized sample covariance
                cov = np.cov(samples, rowvar=0)
                if regular_window > 0:
                    cov = cov * N / (N + regular_window)
                    cov[np.diag_indices_from(cov)] += \
                        regular_variance * regular_window / (N+regular_window)
                potential = quad.QuadPotentialFull(cov)
            else:
                var = np.var(samples, axis=0)
                if regular_window > 0:
                    var = var * N / (N + regular_window)
                    var += \
                        regular_variance * regular_window / (N+regular_window)
                potential = quad.QuadPotentialDiag(cov)

        return pm.NUTS(potential=potential, **kwargs)

    def _extend(self, steps, start=None, step=None, **kwargs):

        kwargs["compute_convergence_checks"] = False
        kwargs["discard_tuned_samples"] = False
        kwargs["draws"] = 2

        # Hide some of the PyMC3 logging
        logger = logging.getLogger("pymc3")
        propagate = logger.propagate
        logger.propagate = False

        self._current_trace = pm.sample(
            start=start, tune=steps, step=step, **kwargs)
        start = [t[-1] for t in self._current_trace._straces.values()]
        self.count += steps

        logger.propagate = propagate

    def warmup(self, start=None, step_kwargs=None, **kwargs):
        """Run an initial warmup phase to find the typical set"""
        if step_kwargs is None:
            step_kwargs = {}
        step = self.get_step_for_trace(**step_kwargs)
        self._extend(self.start, start=start, step=step, **kwargs)
        self._current_step = self.get_step_for_trace(**step_kwargs)
        return self._current_trace

    def _get_start_and_step(self, start=None, step_kwargs=None, trace=None,
                            step=None):
        if step_kwargs is None:
            step_kwargs = {}
        if trace is not None:
            step = self.get_step_for_trace(trace, **step_kwargs)
        else:
            if trace is None:
                trace = self._current_trace
            if step is None:
                step = self._current_step
        if start is None:
            if trace is not None:
                start = [t[-1] for t in trace._straces.values()]
        return start, step

    def extend_tune(self, steps, start=None,
                    step_kwargs=None, trace=None, step=None, **kwargs):
        """Extend the tuning phase by a given number of steps

        After running the sampling, the mass matrix is re-estimated based on
        this run.

        Args:
            steps (int): The number of steps to run.

        """
        if step_kwargs is None:
            step_kwargs = {}
        start, step = self._get_start_and_step(
            start=start, step_kwargs=step_kwargs, trace=trace, step=step)
        self._extend(steps, start=start, step=step, **kwargs)
        self._current_step = self.get_step_for_trace(self._current_trace,
                                                     **step_kwargs)
        return self._current_trace

    def tune(self, tune=1000, start=None, step_kwargs=None, **kwargs):
        """Run the full tuning run for the mass matrix

        This will run ``start`` steps of warmup followed by chains with
        exponentially increasing chains to tune the mass matrix.

        Args:
            tune (int): The total number of steps to run.

        """
        self.count = 0
        self.warmup(start=start, step_kwargs=step_kwargs, **kwargs)
        steps = self.window
        while self.count < tune:
            self.extend_tune(start=start, step_kwargs=step_kwargs, steps=steps,
                             **kwargs)
            steps *= 2
            if self.count + steps + steps*2 > tune:
                steps = tune - self.count
        return self._current_trace

    def sample(self, trace=None, step=None, start=None, step_kwargs=None,
               **kwargs):
        """Run the production sampling using the tuned mass matrix

        This is a light wrapper around ``pymc3.sample`` and any arguments used
        there (for example ``draws``) can be used as input to this method too.

        """
        start, step = self._get_start_and_step(
            start=start, step_kwargs=step_kwargs, trace=trace, step=step)
        kwargs["tune"] = self.finish
        return pm.sample(start=start, step=step, **kwargs)

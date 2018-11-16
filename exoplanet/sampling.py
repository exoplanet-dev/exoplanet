# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["TuningSchedule"]

import logging

import numpy as np

import pymc3 as pm
from pymc3.step_methods.hmc import quadpotential as quad


class TuningSchedule(object):
    # Ref: src/stan/mcmc/windowed_adaptation.hpp in stan repo

    def __init__(self, warmup=1000, start=75, finish=50, window=25,
                 dense=True):
        self.dense = dense

        self.warmup = int(warmup)
        self.start = int(start)
        self.finish = int(finish)
        self.window = int(window)

        if self.start + self.finish + self.window > self.warmup:
            logging.warning("not enough warmup samples for requested "
                            "tuning schedule")
            start = int(0.15 * self.warmup)
            finish = int(0.1 * self.warmup)
            window = int(warmup - start - finish)

        tune = self.warmup - self.start - self.finish
        if self.window + 2 * self.window >= tune:
            windows = [tune]
        else:
            windows = [self.window]
        count = sum(windows)
        while count < tune:
            next_window = windows[-1] * 2
            if count + next_window + 2 * next_window >= tune:
                next_window = tune - count
            windows.append(next_window)
            count += next_window
        self.windows = np.array(windows, dtype=int)

        self._current_step = None
        self._current_trace = None

    def get_step_for_trace(self, trace=None, model=None,
                           regular_window=5, regular_variance=1e-3,
                           **kwargs):
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
                cov = cov * N / (N + regular_window)
                cov[np.diag_indices_from(cov)] += \
                    regular_variance * regular_window / (N + regular_window)
                potential = quad.QuadPotentialFull(cov)
            else:
                var = np.var(samples, axis=0)
                var = var * N / (N + regular_window)
                var += regular_variance * regular_window / (N + regular_window)
                potential = quad.QuadPotentialDiag(cov)

        return pm.NUTS(potential=potential, **kwargs)

    def tune(self, start=None, step_kwargs=None, **kwargs):
        if step_kwargs is None:
            step_kwargs = {}

        kwargs["compute_convergence_checks"] = False
        kwargs["discard_tuned_samples"] = False
        kwargs["draws"] = 2

        # Start by running the startup
        logging.info("running start-up...")
        step = self.get_step_for_trace(**step_kwargs)

        burnin_trace = pm.sample(
            start=start, tune=self.start, step=step, **kwargs)
        start = [t[-1] for t in burnin_trace._straces.values()]

        for i, steps in enumerate(self.windows):
            logging.info("running adaptation {0} / {1}..."
                         .format(i+1, len(self.windows)))
            step = self.get_step_for_trace(burnin_trace, **step_kwargs)
            burnin_trace = pm.sample(
                start=start, tune=steps, step=step, **kwargs)
            start = [t[-1] for t in burnin_trace._straces.values()]

        self._current_trace = burnin_trace
        self._current_step = self.get_step_for_trace(burnin_trace,
                                                     **step_kwargs)
        return self._current_trace

    def sample(self, trace=None, step=None, start=None, step_kwargs=None,
               **kwargs):
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

        kwargs["tune"] = self.finish
        return pm.sample(start=start, step=step, **kwargs)

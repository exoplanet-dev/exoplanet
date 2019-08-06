#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys

if len(sys.argv) > 1:
    N_pl = int(sys.argv[1])
else:
    N_pl = 1
dirname = "{0:02d}".format(N_pl)
if len(sys.argv) > 2:
    version = int(sys.argv[2])
    dirname = os.path.join(dirname, "{0:04d}".format(version))
else:
    version = 0
os.makedirs(dirname, exist_ok=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["THEANO_FLAGS"] = "compiledir=./{0}/cache".format(dirname)

import h5py  # NOQA
import time  # NOQA
import emcee  # NOQA
import string  # NOQA
import numpy as np  # NOQA
import pymc3 as pm  # NOQA
import matplotlib.pyplot as plt  # NOQA

# NOQA
import theano  # NOQA
import theano.tensor as tt  # NOQA

# NOQA
import exoplanet as xo  # NOQA

# Parameters
target_n_eff = 2000
np.random.seed(1234 + version)

# Simulate the planet properties
periods = np.exp(np.random.uniform(np.log(1), np.log(100), size=N_pl))
t0s = periods * np.random.rand(N_pl)
Ks = np.sort(np.exp(np.random.uniform(np.log(2), np.log(10), N_pl)))[::-1]
eccs = np.random.uniform(0.25, 0.5, N_pl)
omegas = np.random.uniform(-np.pi, np.pi, N_pl)

# Simulate the time sampling
N = 25 + 25 * N_pl
x = np.sort(np.random.uniform(-2 * 365, 2 * 365, N))
yerr = np.random.uniform(0.5, 5.0, N)
t = np.linspace(x.min() - 5, x.max() + 5, 1000)

# Define the model
with pm.Model() as model:

    # Gaussian priors based on transit data (from Petigura et al.)
    t0 = pm.Normal("t0", mu=np.array(t0s), sd=10.0, shape=N_pl)
    logP = pm.Normal("logP", mu=np.log(periods), sd=1.0, shape=N_pl)
    P = pm.Deterministic("P", tt.exp(logP))

    # Wide log-normal prior for semi-amplitude
    logK = pm.Normal("logK", mu=np.log(Ks), sd=10.0, shape=N_pl)

    # This is a sanity check that restricts the semiamplitude to reasonable
    # values because things can get ugly as K -> 0
    pm.Potential("logK_bound", tt.switch(logK < -2.0, -np.inf, 0.0))

    # The amlitudes should be sorted
    pm.Potential("logK_order", tt.switch(logK[1:] > logK[:-1], -np.inf, 0.0))

    # We also want to keep period physical but this probably won't be hit
    pm.Potential("P_bound", tt.switch(P <= 0, -np.inf, 0.0))

    # Eccentricity & argument of periasteron
    ecc = pm.Uniform("ecc", lower=0, upper=0.99, shape=N_pl, testval=eccs)
    omega = xo.distributions.Angle("omega", shape=N_pl, testval=omegas)

    # Jitter & a quadratic RV trend
    # logs = pm.Normal("logs", mu=np.log(np.median(yerr)), sd=5.0)
    trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3)

    # Set up the orbit
    orbit = xo.orbits.KeplerianOrbit(period=P, t0=t0, ecc=ecc, omega=omega)

    # Set up the RV model and save it as a deterministic
    # for plotting purposes later
    vrad = orbit.get_radial_velocity(x, K=tt.exp(logK))
    if N_pl == 1:
        vrad = vrad[:, None]

    # Define the background model
    A = np.vander(x - 0.5 * (x.min() + x.max()), 3)
    bkg = tt.dot(A, trend)

    # Sum over planets and add the background to get the full model
    rv_model = tt.sum(vrad, axis=-1) + bkg

    # Simulate the data
    y_true = xo.eval_in_model(rv_model)
    y = y_true + yerr * np.random.randn(len(yerr))

    # Compute the prediction
    vrad_pred = orbit.get_radial_velocity(t, K=tt.exp(logK))
    if N_pl == 1:
        vrad_pred = vrad_pred[:, None]
    A_pred = np.vander(t - 0.5 * (x.min() + x.max()), 3)
    bkg_pred = tt.dot(A_pred, trend)
    rv_model_pred = tt.sum(vrad_pred, axis=-1) + bkg_pred

    # Likelihood
    err = yerr
    # err = tt.sqrt(yerr**2 + tt.exp(2*logs))
    pm.Normal("obs", mu=rv_model, sd=err, observed=y)

    # Optimize
    map_soln = pm.find_MAP(start=model.test_point, vars=[trend])
    map_soln = pm.find_MAP(start=map_soln)


def check_convergence(samples):
    tau = emcee.autocorr.integrated_time(samples, tol=0)
    num = samples.shape[0] * samples.shape[1]
    converged = np.all(tau * target_n_eff < num)
    converged &= np.all(len(samples) > 50 * tau)
    return converged, num / tau


# Run the PyMC3 sampler
chains = 2
sampler = xo.PyMC3Sampler(start=500, finish=500, window=500)
with model:
    burnin = sampler.tune(
        tune=100000, start=map_soln, chains=chains, cores=1, progressbar=False
    )

tottime = 0
trace = None
with model:
    while True:
        strt = time.time()
        trace = sampler.sample(
            draws=2000, trace=trace, chains=chains, cores=1, progressbar=False
        )
        tottime += time.time() - strt

        samples = np.array(trace.get_values("P", combine=False))
        samples = np.moveaxis(samples, 0, 1)
        flag, n_eff = check_convergence(samples)
        if flag:
            break

    time_pymc = tottime
    time_ind_pymc = tottime / n_eff
    n_eff_pymc = n_eff

# Save the trace file
df = pm.trace_to_dataframe(trace)
df.to_hdf(os.path.join(dirname, "pymc-trace.h5"), "trace")

# Make the plots
for n, letter in enumerate(string.ascii_lowercase[1 : N_pl + 1]):
    fig = plt.figure()

    # Get the posterior median orbital parameters
    p = np.median(trace["P"][:, n])
    t0 = np.median(trace["t0"][:, n])

    # Plot the folded data
    x_fold = (x - t0 + 0.5 * p) % p - 0.5 * p
    plt.errorbar(x_fold, y, yerr=yerr, fmt=".k")

    plt.annotate(
        "period = {0:.4f} +/- {1:.4f} d".format(p, np.std(trace["P"][:, n])),
        (0, 1),
        xycoords="axes fraction",
        xytext=(5, -5),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=12,
    )

    plt.annotate(
        "true period = {0:.4f} d".format(periods[n]),
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=12,
    )

    plt.legend(fontsize=10)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("phase [days]")
    plt.ylabel("radial velocity [m/s]")

    fig.savefig(
        os.path.join(dirname, "phase-{0}.pdf".format(letter)),
        bbox_inches="tight",
    )
    plt.close(fig)


# Fit using emcee
with model:
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

    # Build a wrapper around the theano model
    f = theano.function(
        model.vars, [model.logpt] + model.vars + model.deterministics
    )

    def log_prob_func(params):
        dct = model.bijection.rmap(params)
        args = (dct[k.name] for k in model.vars)
        results = f(*args)
        return tuple(results)

    # First we work out the shapes of all of the deterministic variables
    res = model.test_point
    vec = model.bijection.map(res)
    initial_blobs = log_prob_func(vec)[1:]
    dtype = [
        (var.name, float, np.shape(b))
        for var, b in zip(model.vars + model.deterministics, initial_blobs)
    ]

    # Then sample as usual
    coords = samples[
        np.random.randint(len(samples), size=2 * samples.shape[1])
    ]
    nwalkers, ndim = coords.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_func, blobs_dtype=dtype
    )
    thin_by = 10
    tottime = 0
    for i in range(1000):
        strt = time.time()
        sampler.run_mcmc(coords, 500, thin_by=thin_by, progress=False)
        tottime += time.time() - strt

        samples = sampler.get_blobs()["P"]
        flag, n_eff = check_convergence(samples)
        if flag:
            break

    time_emcee = tottime
    time_ind_emcee = tottime / n_eff
    n_eff_emcee = n_eff

blobs = sampler.get_blobs()
with h5py.File(os.path.join(dirname, "emcee-trace.h5"), "w") as f:
    f.create_dataset("trace", data=blobs)

print("time per ind. sample, emcee: {0}".format(time_ind_emcee))
print("time per ind. sample, pymc: {0}".format(time_ind_pymc))
print(
    "time per ind. sample, ratio: {0}".format(time_ind_emcee / time_ind_pymc)
)
ndim = model.ndim
with open(os.path.join(dirname, "results.csv"), "w") as f:
    f.write("method,ndim,time\n")
    for line in time_ind_emcee:
        f.write("emcee,{0},{1}\n".format(ndim, line))
    for line in time_ind_pymc:
        f.write("pymc,{0},{1}\n".format(ndim, line))

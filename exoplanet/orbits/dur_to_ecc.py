# -*- coding: utf-8 -*-

__all__ = ["duration_to_eccentricity"]

from itertools import product

import numpy as np
import pymc3 as pm
import theano.tensor as tt

from .keplerian import KeplerianOrbit, _get_consistent_inputs


def duration_to_eccentricity(func, duration, ror, **kwargs):
    num_planets = kwargs.pop("num_planets", 1)
    orbit_type = kwargs.pop("orbit_type", KeplerianOrbit)
    name = kwargs.get("name", "dur_ecc")

    inputs = _get_consistent_inputs(
        kwargs.get("a", None),
        kwargs.get("period", None),
        kwargs.get("rho_star", None),
        kwargs.get("r_star", None),
        kwargs.get("m_star", None),
        kwargs.get("rho_star_units", None),
        kwargs.get("m_planet", 0.0),
        kwargs.get("m_planet_units", None),
    )
    a, period, rho_star, r_star, m_star, m_planet = inputs
    b = kwargs.get("b", 0.0)
    s = tt.sin(kwargs["omega"])
    umax_inv = tt.switch(tt.lt(s, 0), tt.sqrt(1 - s ** 2), 1.0)

    const = (
        period * tt.shape_padright(r_star) * tt.sqrt((1 + ror) ** 2 - b ** 2)
    )
    const /= np.pi * a

    u = duration / const

    e1 = -s * u ** 2 / ((s * u) ** 2 + 1)
    e2 = tt.sqrt((s ** 2 - 1) * u ** 2 + 1) / ((s * u) ** 2 + 1)

    models = []
    logjacs = []
    logprobs = []
    for args in product(*(zip("np", (-1, 1)) for _ in range(num_planets))):
        labels, signs = zip(*args)

        # Compute the eccentricity branch
        ecc = tt.stack([e1[i] + signs[i] * e2[i] for i in range(num_planets)])

        # Work out the Jacobian
        valid_ecc = tt.and_(tt.lt(ecc, 1.0), tt.ge(ecc, 0.0))
        logjac = tt.switch(
            tt.all(valid_ecc),
            tt.sum(
                0.5 * tt.log(1 - ecc ** 2)
                + 2 * tt.log(s * ecc + 1)
                - tt.log(tt.abs_(s + ecc))
                - tt.log(const)
            ),
            -np.inf,
        )
        ecc = tt.switch(valid_ecc, ecc, tt.zeros_like(ecc))

        # Create a sub-model to capture this component
        with pm.Model(name="dur_ecc_" + "_".join(labels)) as model:
            pm.Deterministic("ecc", ecc)
            orbit = orbit_type(ecc=ecc, **kwargs)
            logprob = tt.sum(func(orbit))

        models.append(model)
        logjacs.append(logjac)
        logprobs.append(logprob)

    # Compute the marginalized likelihood
    logjacs = tt.stack(logjacs)
    logprobs = tt.stack(logprobs)
    logprob = tt.switch(
        tt.gt(1.0 / u, umax_inv),
        tt.sum(pm.logsumexp(logprobs + logjacs)),
        -np.inf,
    )
    pm.Potential(name + "_logp", logprob)
    pm.Deterministic(name + "_logjacs", logjacs)
    pm.Deterministic(name + "_logprobs", logprobs)

    # Loop over models and compute the marginalized values for all the
    # parameters and deterministics
    norm = tt.sum(pm.logsumexp(logjacs))
    logw = tt.switch(
        tt.gt(1.0 / u, umax_inv),
        logjacs - norm,
        -np.inf + tt.zeros_like(logjacs),
    )
    pm.Deterministic(name + "_logw", logw)
    for k in models[0].named_vars.keys():
        name = k[len(models[0].name) + 1 :]
        pm.Deterministic(
            name,
            sum(
                tt.exp(logw[i]) * model.named_vars[model.name + "_" + name]
                for i, model in enumerate(models)
            ),
        )

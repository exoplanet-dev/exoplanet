"""This module has an EXPERIMENTAL and UNTESTED implementation of analytic
marginalizion over eccentricity as a function of transit duration (since the
mapping is not 1-to-1). The implementation as written probably DOES NOT WORK,
but I've (DFM) left it here for my own future reference. Feel free to ping me
if you're interested in what's going on here!"""

__all__ = ["duration_to_eccentricity"]

from itertools import product

import numpy as np

from exoplanet.compat import pm
from exoplanet.compat import tensor as pt
from exoplanet.orbits.keplerian import KeplerianOrbit, _get_consistent_inputs


def duration_to_eccentricity(
    func, duration, ror, **kwargs
):  # pragma: no cover
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
    s = pt.sin(kwargs["omega"])
    umax_inv = pt.switch(pt.lt(s, 0), pt.sqrt(1 - s**2), 1.0)

    const = (
        period * pt.shape_padright(r_star) * pt.sqrt((1 + ror) ** 2 - b**2)
    )
    const /= np.pi * a

    u = duration / const

    e1 = -s * u**2 / ((s * u) ** 2 + 1)
    e2 = pt.sqrt((s**2 - 1) * u**2 + 1) / ((s * u) ** 2 + 1)

    models = []
    logjacs = []
    logprobs = []
    for args in product(*(zip("np", (-1, 1)) for _ in range(num_planets))):
        labels, signs = zip(*args)

        # Compute the eccentricity branch
        ecc = pt.stack([e1[i] + signs[i] * e2[i] for i in range(num_planets)])

        # Work out the Jacobian
        valid_ecc = pt.and_(pt.lt(ecc, 1.0), pt.ge(ecc, 0.0))
        logjac = pt.switch(
            pt.all(valid_ecc),
            pt.sum(
                0.5 * pt.log(1 - ecc**2)
                + 2 * pt.log(s * ecc + 1)
                - pt.log(pt.abs(s + ecc))
                - pt.log(const)
            ),
            -np.inf,
        )
        ecc = pt.switch(valid_ecc, ecc, pt.zeros_like(ecc))

        # Create a sub-model to capture this component
        with pm.Model(name="dur_ecc_" + "_".join(labels)) as model:
            pm.Deterministic("ecc", ecc)
            orbit = orbit_type(ecc=ecc, **kwargs)
            logprob = pt.sum(func(orbit))

        models.append(model)
        logjacs.append(logjac)
        logprobs.append(logprob)

    # Compute the marginalized likelihood
    logjacs = pt.stack(logjacs)
    logprobs = pt.stack(logprobs)
    logprob = pt.switch(
        pt.gt(1.0 / u, umax_inv),
        pt.sum(pm.logsumexp(logprobs + logjacs)),
        -np.inf,
    )
    pm.Potential(name + "_logp", logprob)
    pm.Deterministic(name + "_logjacs", logjacs)
    pm.Deterministic(name + "_logprobs", logprobs)

    # Loop over models and compute the marginalized values for all the
    # parameters and deterministics
    norm = pt.sum(pm.logsumexp(logjacs))
    logw = pt.switch(
        pt.gt(1.0 / u, umax_inv),
        logjacs - norm,
        -np.inf + pt.zeros_like(logjacs),
    )
    pm.Deterministic(name + "_logw", logw)
    for k in models[0].named_vars.keys():
        name = k[len(models[0].name) + 1 :]
        pm.Deterministic(
            name,
            sum(
                pt.exp(logw[i]) * model.named_vars[model.name + "_" + name]
                for i, model in enumerate(models)
            ),
        )

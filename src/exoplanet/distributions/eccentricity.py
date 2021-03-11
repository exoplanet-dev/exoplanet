# -*- coding: utf-8 -*-

__all__ = ["kipping13", "vaneylen19"]

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm

from ..citations import add_citations_to_model
from .base import UnitUniform


def kipping13(
    name, fixed=False, long=None, lower=None, upper=None, model=None, **kwargs
):
    """The beta eccentricity distribution fit by Kipping (2013)

    The beta distribution parameters fit by `Kipping (2013b)
    <https://arxiv.org/abs/1306.4982>`_.

    Args:
        name (str): The name of the eccentricity variable.
        fixed (bool, optional): If ``True``, use the posterior median
            hyperparameters. Otherwise, marginalize over the parameters.
        long (bool, optional): If ``True``, use the parameters for the long
            period fit. If ``False``, use the parameters for the short period
            fit. If not given, the parameters fit using the full dataset are
            used.
        lower (float, optional): Restrict the eccentricity to be larger than
            this value.
        upper (float, optional): Restrict the eccentricity to be smaller than
            this value.

    Returns:
        The eccentricity distribution.

    """
    model = pm.modelcontext(model)
    add_citations_to_model(["kipping13b"], model=model)

    if long is None:
        # If 'long' is not provided, use the fit for the parameters from the
        # full dataset
        alpha_mu = 1.12
        alpha_sd = 0.1
        beta_mu = 3.09
        beta_sd = 0.3
    else:
        # If 'long' is set, select either the long or short period model
        # parameters
        if long:
            alpha_mu = 1.12
            alpha_sd = 0.1
            beta_mu = 3.09
            beta_sd = 0.3
        else:
            alpha_mu = 0.697
            alpha_sd = 0.4
            beta_mu = 3.27
            beta_sd = 0.3

    with model:
        if fixed:
            # Use the posterior median parameters
            alpha = alpha_mu
            beta = beta_mu
        else:
            # Marginalize over the uncertainty on the parameters of the beta
            with pm.Model(name=name):
                bounded_normal = pm.Bound(pm.Normal, lower=0)
                alpha = bounded_normal(
                    "alpha", mu=alpha_mu, sd=alpha_sd, testval=alpha_mu
                )
                beta = bounded_normal(
                    "beta", mu=beta_mu, sd=beta_sd, testval=beta_mu
                )

        # Allow for upper and lower bounds
        if lower is not None or upper is not None:
            dist = pm.Bound(
                pm.Beta,
                lower=0.0 if lower is None else lower,
                upper=1.0 if upper is None else upper,
            )
            return dist(name, alpha=alpha, beta=beta, **kwargs)

        return pm.Beta(name, alpha=alpha, beta=beta, **kwargs)


def vaneylen19(
    name,
    fixed=False,
    multi=False,
    lower=None,
    upper=None,
    model=None,
    **kwargs,
):
    """The eccentricity distribution for small planets

    The mixture distribution fit by `Van Eylen et al. (2019)
    <https://arxiv.org/abs/1807.00549>`_ to a population of well-characterized
    small transiting planets observed by Kepler.

    Args:
        name (str): The name of the eccentricity variable.
        fixed (bool, optional): If ``True``, use the posterior median
            hyperparameters. Otherwise, marginalize over the parameters.
        multi (bool, optional): If ``True``, use the distribution for systems
            with multiple transiting planets. If ``False`` (default), use the
            distribution for systems with only one detected transiting planet.
        lower (float, optional): Restrict the eccentricity to be larger than
            this value.
        upper (float, optional): Restrict the eccentricity to be smaller than
            this value.

    Returns:
        The eccentricity distribution.

    """

    model = pm.modelcontext(model)
    add_citations_to_model(["vaneylen19"], model=model)

    sigma_gauss_mu = 0.049
    sigma_gauss_sd = 0.02
    sigma_rayleigh_mu = 0.26
    sigma_rayleigh_sd = 0.05
    if multi:
        frac_mu = 0.08
        frac_sd = 0.08
    else:
        frac_mu = 0.76
        frac_sd = 0.2

    with model:
        if lower is None and upper is None:
            ecc = UnitUniform(name, **kwargs)
        else:
            ecc = pm.Uniform(
                name,
                lower=0.0 if lower is None else lower,
                upper=1.0 if upper is None else upper,
                **kwargs,
            )

        with pm.Model(name=name):

            if fixed:
                sigma_gauss = sigma_gauss_mu
                sigma_rayleigh = sigma_rayleigh_mu
                frac = frac_mu
            else:

                bounded_normal = pm.Bound(pm.Normal, lower=0)
                sigma_gauss = bounded_normal(
                    "sigma_gauss",
                    mu=sigma_gauss_mu,
                    sd=sigma_gauss_sd,
                    testval=sigma_gauss_mu,
                )
                sigma_rayleigh = bounded_normal(
                    "sigma_rayleigh",
                    mu=sigma_rayleigh_mu,
                    sd=sigma_rayleigh_sd,
                    testval=sigma_rayleigh_mu,
                )
                frac = pm.Bound(pm.Normal, lower=0, upper=1)(
                    "frac", mu=frac_mu, sd=frac_sd, testval=frac_mu
                )

            gauss = pm.HalfNormal.dist(sigma=sigma_gauss)
            rayleigh = pm.Weibull.dist(
                alpha=2, beta=np.sqrt(2) * sigma_rayleigh
            )

            pm.Potential(
                "prior",
                pm.math.logaddexp(
                    tt.log(1 - frac) + gauss.logp(ecc),
                    tt.log(frac) + rayleigh.logp(ecc),
                ),
            )

        return ecc

__all__ = ["kipping13", "vaneylen19"]

import numpy as np

from exoplanet.citations import add_citations_to_model
from exoplanet.compat import USING_PYMC3, pm
from exoplanet.compat import tensor as pt


def kipping13(
    name, fixed=True, long=None, lower=None, upper=None, model=None, **kwargs
):
    """The beta distribution parameters fit by `Kipping (2013b)
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
                alpha = _truncate(
                    "alpha",
                    pm.Normal,
                    lower=0,
                    mu=alpha_mu,
                    sigma=alpha_sd,
                    **_with_initval(initval=alpha_mu),
                )
                beta = _truncate(
                    "beta",
                    pm.Normal,
                    lower=0,
                    mu=beta_mu,
                    sigma=beta_sd,
                    **_with_initval(initval=beta_mu),
                )

        # Allow for upper and lower bounds
        ecc = kwargs.pop("observed", None)
        if lower is not None or upper is not None:
            lower = 0.0 if lower is None else lower
            upper = 1.0 if upper is None else upper
            kwargs["initval"] = kwargs.pop(
                "initval", kwargs.pop("testval", 0.5 * (lower + upper))
            )
            if ecc is None:
                return _truncate(
                    name,
                    pm.Beta,
                    lower=lower,
                    upper=upper,
                    alpha=alpha,
                    beta=beta,
                    **_with_initval(**kwargs),
                )
            else:
                dist = _truncate_dist(
                    pm.Beta,
                    lower=lower,
                    upper=upper,
                    alpha=alpha,
                    beta=beta,
                    **_with_initval(**kwargs),
                )
        else:
            if ecc is None:
                return pm.Beta(name, alpha=alpha, beta=beta, **kwargs)
            else:
                dist = pm.Beta.dist(alpha=alpha, beta=beta, **kwargs)

        return pm.Potential(name, _logp(dist, ecc))


def vaneylen19(
    name,
    fixed=True,
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
                sigma_gauss = _truncate(
                    "sigma_gauss",
                    pm.Normal,
                    lower=0,
                    mu=sigma_gauss_mu,
                    sigma=sigma_gauss_sd,
                    **_with_initval(initval=sigma_gauss_mu),
                )
                sigma_rayleigh = _truncate(
                    "sigma_rayleigh",
                    pm.Normal,
                    lower=0,
                    mu=sigma_rayleigh_mu,
                    sigma=sigma_rayleigh_sd,
                    **_with_initval(initval=sigma_rayleigh_mu),
                )
                frac = _truncate(
                    "frac",
                    pm.Normal,
                    lower=0,
                    upper=1,
                    mu=frac_mu,
                    sigma=frac_sd,
                    **_with_initval(initval=frac_mu),
                )

            gauss = pm.HalfNormal.dist(sigma=sigma_gauss)
            rayleigh = pm.Weibull.dist(
                alpha=2, beta=np.sqrt(2) * sigma_rayleigh
            )

            pm.Potential(
                "prior",
                pm.math.logaddexp(
                    pt.log(1 - frac) + _logp(gauss, ecc),
                    pt.log(frac) + _logp(rayleigh, ecc),
                ),
            )

        return ecc


def _with_initval(**kwargs):
    val = kwargs.pop("initval", kwargs.pop("testval", None))
    if USING_PYMC3:
        return dict(kwargs, testval=val)
    return dict(kwargs, initval=val)


def _logp(rv, x):
    if USING_PYMC3:
        return rv.logp(x)
    return pm.logp(rv, x)


def _truncate(name, dist, *, lower=None, upper=None, **kwargs):
    if USING_PYMC3:
        return pm.Bound(dist, lower=lower, upper=upper)(
            name, **_with_initval(**kwargs)
        )
    initval = kwargs.pop("initval", kwargs.pop("testval", None))
    return pm.Truncated(
        name, dist.dist(**kwargs), lower=lower, upper=upper, initval=initval
    )

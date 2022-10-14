__all__ = [
    "angle",
    "unit_disk",
    "quad_limb_dark",
    "impact_parameter",
    "kipping13",
    "vaneylen19",
]

import numpy as np

from exoplanet.compat import pm, tensor as at, USING_PYMC3
from exoplanet.citations import add_citations_to_model


def with_initval(val, **kwargs):
    if USING_PYMC3:
        return dict(kwargs, testval=val)
    return dict(kwargs, initval=val)


def angle(name, *, regularization=10.0, **kwargs):
    initval = kwargs.pop("initval", kwargs.pop("testval", 0.0))
    x1 = pm.Normal(
        f"{name}_angle1__", **with_initval(np.sin(initval), **kwargs)
    )
    x2 = pm.Normal(
        f"{name}_angle2__", **with_initval(np.cos(initval), **kwargs)
    )
    if regularization is not None:
        pm.Potential(
            f"_{name}_regularization",
            regularization * at.log(x1**2 + x2**2),
        )
    return pm.Deterministic(name, at.arctan2(x1, x2))


def unit_disk(name_x, name_y, **kwargs):
    initval = kwargs.pop("initval", kwargs.pop("testval", [0.0, 0.0]))
    kwargs["lower"] = -1.0
    kwargs["upper"] = 1.0
    x1 = pm.Uniform(name_x, **with_initval(initval[0], **kwargs))
    x2 = pm.Uniform(
        f"{name_y}_unit_disk__",
        **with_initval(initval[1] * np.sqrt(1 - initval[0] ** 2), **kwargs),
    )
    norm = at.sqrt(1 - x1**2)
    pm.Potential(f"{name_y}_jacobian", at.log(norm))
    return x1, pm.Deterministic(name_y, x2 * norm)


def quad_limb_dark(name, **kwargs):
    u = kwargs.pop("initval", kwargs.pop("testval", [np.sqrt(0.5), 0.0]))
    u1 = u[0]
    u2 = u[1]
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    q1 = pm.Uniform(f"{name}_q1__", **with_initval((u1 + u2) ** 2, **kwargs))
    q2 = pm.Uniform(
        f"{name}_q2__", **with_initval(0.5 * u1 / (u1 + u2), **kwargs)
    )
    sqrtq1 = at.sqrt(q1)
    twoq2 = 2 * q2
    return pm.Deterministic(
        name, at.stack([sqrtq1 * twoq2, sqrtq1 * (1 - twoq2)], axis=0)
    )


def impact_parameter(name, ror, **kwargs):
    b = kwargs.pop("initval", kwargs.pop("testval", 0.5))
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    norm = pm.Uniform(
        f"{name}_impact_parameter__", **with_initval(b / (1 + ror), **kwargs)
    )
    return pm.Deterministic(name, norm * (1 + ror))


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
                    "alpha", mu=alpha_mu, sd=alpha_sd, **with_initval(alpha_mu)
                )
                beta = bounded_normal(
                    "beta", mu=beta_mu, sd=beta_sd, **with_initval(beta_mu)
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
                    **with_initval(sigma_gauss_mu),
                )
                sigma_rayleigh = bounded_normal(
                    "sigma_rayleigh",
                    mu=sigma_rayleigh_mu,
                    sd=sigma_rayleigh_sd,
                    **with_initval(sigma_rayleigh_mu),
                )
                frac = pm.Bound(pm.Normal, lower=0, upper=1)(
                    "frac", mu=frac_mu, sd=frac_sd, **with_initval(frac_mu)
                )

            gauss = pm.HalfNormal.dist(sigma=sigma_gauss)
            rayleigh = pm.Weibull.dist(
                alpha=2, beta=np.sqrt(2) * sigma_rayleigh
            )

            pm.Potential(
                "prior",
                pm.math.logaddexp(
                    at.log(1 - frac) + gauss.logp(ecc),
                    at.log(frac) + rayleigh.logp(ecc),
                ),
            )

        return ecc

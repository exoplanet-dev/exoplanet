__all__ = ["angle", "unit_disk", "quad_limb_dark", "impact_parameter"]

import numpy as np

from exoplanet.citations import add_citations_to_model
from exoplanet.compat import USING_PYMC3, pm
from exoplanet.compat import tensor as at
from exoplanet.utils import as_tensor_variable


def angle(name, *, regularization=10.0, **kwargs):
    initval = kwargs.pop("initval", kwargs.pop("testval", 0.0))
    x1 = pm.Normal(
        f"{name}_angle1__", **_with_initval(np.sin(initval), **kwargs)
    )
    x2 = pm.Normal(
        f"{name}_angle2__", **_with_initval(np.cos(initval), **kwargs)
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
    x1 = pm.Uniform(name_x, **_with_initval(initval[0], **kwargs))
    x2 = pm.Uniform(
        f"{name_y}_unit_disk__",
        **_with_initval(initval[1] * np.sqrt(1 - initval[0] ** 2), **kwargs),
    )
    norm = at.sqrt(1 - x1**2)
    pm.Potential(f"{name_y}_jacobian", at.log(norm))
    return x1, pm.Deterministic(name_y, x2 * norm)


def quad_limb_dark(name, **kwargs):
    add_citations_to_model(("kipping13",), kwargs.get("model", None))

    u = kwargs.pop("initval", kwargs.pop("testval", [np.sqrt(0.5), 0.0]))
    u1 = u[0]
    u2 = u[1]
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    q1 = pm.Uniform(f"{name}_q1__", **_with_initval((u1 + u2) ** 2, **kwargs))
    q2 = pm.Uniform(
        f"{name}_q2__", **_with_initval(0.5 * u1 / (u1 + u2), **kwargs)
    )
    sqrtq1 = at.sqrt(q1)
    twoq2 = 2 * q2
    return pm.Deterministic(
        name, at.stack([sqrtq1 * twoq2, sqrtq1 * (1 - twoq2)], axis=0)
    )


def impact_parameter(name, ror, **kwargs):
    ror = as_tensor_variable(ror)
    bhat = kwargs.pop("initval", kwargs.pop("testval", 0.5))
    if not USING_PYMC3:
        shape = kwargs.setdefault("shape", ror.shape)
        bhat = at.broadcast_to(bhat, shape)
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    norm = pm.Uniform(
        f"{name}_impact_parameter__",
        **_with_initval(bhat / (1 + ror), **kwargs),
    )
    return pm.Deterministic(name, norm * (1 + ror))


def _with_initval(val, **kwargs):
    if USING_PYMC3:
        return dict(kwargs, testval=val)
    return dict(kwargs, initval=val)

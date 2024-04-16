__all__ = ["angle", "unit_disk", "quad_limb_dark", "impact_parameter"]

import numpy as np

from exoplanet.citations import add_citations_to_model
from exoplanet.compat import USING_PYMC3, pm
from exoplanet.compat import tensor as pt
from exoplanet.utils import as_tensor_variable


def angle(name, *, regularization=10.0, **kwargs):
    """An angle constrained to be in the range -pi to pi

    The actual sampling is performed in the two dimensional vector space
    proportional to ``(sin(theta), cos(theta))`` so that the sampler doesn't see
    a discontinuity at pi.

    The regularization parameter can be used to improve sampling performance
    when the value of the angle is well constrained. It removes prior mass near
    the origin in the sampling space, which can lead to bad geometry when the
    angle is poorly constrained, but better performance when it is. The default
    value of ``10.0`` is a good starting point.
    """
    shape_arg = kwargs.get("shape", ())
    initval = kwargs.pop("initval", kwargs.pop("testval", np.zeros(shape_arg)))
    x1 = pm.Normal(
        f"__{name}_angle1", **_with_initval(initval=np.sin(initval), **kwargs)
    )
    x2 = pm.Normal(
        f"__{name}_angle2", **_with_initval(initval=np.cos(initval), **kwargs)
    )
    if regularization is not None:
        pm.Potential(
            f"{name}_regularization",
            regularization * pt.log(x1**2 + x2**2),
        )
    return pm.Deterministic(name, pt.arctan2(x1, x2))


def unit_disk(name_x, name_y, **kwargs):
    """Two dimensional parameters constrained to live within the unit disk

    This returns two distributions whose sum of squares will be in the range
    ``[0, 1)``. For example, in this code block:

    .. code-block:: python

        x, y = unit_disk("x", "y") radius_sq = x**2 + y**2

    the tensor ``radius_sq`` will always have a value in the range ``[0, 1)``.

    Args:
        name_x: The name of the first distribution.
        name_y: The name of the second distribution.
    """
    shape_kwarg = kwargs.get("shape", ())
    if isinstance(shape_kwarg, int):
        shape_kwarg = (shape_kwarg,)
    init_shape = (2,) + shape_kwarg
    initval = kwargs.pop(
        "initval", kwargs.pop("testval", np.zeros(init_shape))
    )
    # initval = kwargs.pop("initval", kwargs.pop("testval", [0.0, 0.0]))
    kwargs["lower"] = -1.0
    kwargs["upper"] = 1.0
    x1 = pm.Uniform(name_x, **_with_initval(initval=initval[0], **kwargs))
    x2 = pm.Uniform(
        f"__{name_y}_unit_disk",
        **_with_initval(
            initval=initval[1] * np.sqrt(1 - initval[0] ** 2), **kwargs
        ),
    )
    norm = pt.sqrt(1 - x1**2)
    pm.Potential(f"{name_y}_jacobian", pt.log(norm))
    return x1, pm.Deterministic(name_y, x2 * norm)


def quad_limb_dark(name, **kwargs):
    """An uninformative prior for quadratic limb darkening parameters

    This is an implementation of the `Kipping (2013)
    <https://arxiv.org/abs/1308.0009>`_ reparameterization of the two-parameter
    limb darkening model to allow for efficient and uninformative sampling.
    """
    add_citations_to_model(("kipping13",), kwargs.get("model", None))

    u = kwargs.pop("initval", kwargs.pop("testval", [np.sqrt(0.5), 0.0]))
    u1 = u[0]
    u2 = u[1]
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    q1 = pm.Uniform(
        f"__{name}_q1", **_with_initval(initval=(u1 + u2) ** 2, **kwargs)
    )
    q2 = pm.Uniform(
        f"__{name}_q2", **_with_initval(initval=0.5 * u1 / (u1 + u2), **kwargs)
    )
    sqrtq1 = pt.sqrt(q1)
    twoq2 = 2 * q2
    return pm.Deterministic(
        name, pt.stack([sqrtq1 * twoq2, sqrtq1 * (1 - twoq2)], axis=0)
    )


def impact_parameter(name, ror, **kwargs):
    """The impact parameter distribution for a transiting planet

    Args:
        ror: A scalar, tensor, or PyMC (3 or 5+) distribution representing the radius
            ratio between the planet and star. Conditioned on a value of
            ``ror``, this will be uniformly distributed between ``0`` and
            ``1+ror``.
    """
    ror = as_tensor_variable(ror)
    bhat = kwargs.pop("initval", kwargs.pop("testval", 0.5))
    if not USING_PYMC3:
        shape = kwargs.setdefault("shape", ror.shape)
        if isinstance(shape, int):
            shape = (shape,)
        bhat = pt.broadcast_to(bhat, shape)
    kwargs["lower"] = 0.0
    kwargs["upper"] = 1.0
    norm = pm.Uniform(
        f"__{name}_impact_parameter",
        **_with_initval(initval=bhat / (1 + ror), **kwargs),
    )
    return pm.Deterministic(name, norm * (1 + ror))


def _with_initval(**kwargs):
    val = kwargs.pop("initval", kwargs.pop("testval", None))
    if USING_PYMC3:
        return dict(kwargs, testval=val)
    return dict(kwargs, initval=val)

# -*- coding: utf-8 -*-

__all__ = ["kepler"]
from functools import partial

import numpy as np
from jax import core
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from . import xla_ops

xops = xla_client.ops

xla_client.register_cpu_custom_call_target(b"kepler", xla_ops.kepler())


def kepler(M, ecc):
    return kepler_p.bind(M, ecc)


def kepler_abstract_eval(M, ecc):
    if M.dtype != jnp.float64 or ecc.dtype != jnp.float64:
        raise ValueError("double precision is required")
    if M.shape != ecc.shape:
        raise ValueError("dimension mismatch")
    return (ShapedArray(M.shape, M.dtype), ShapedArray(M.shape, M.dtype))


def kepler_translation_rule(c, M, ecc):
    M_shape = c.get_shape(M)
    ecc_shape = c.get_shape(ecc)
    if M_shape != ecc_shape:
        raise ValueError("dimension mismatch")

    N = np.prod(np.asarray(M_shape.dimensions(), dtype=np.int32))
    return xops.CustomCallWithLayout(
        c,
        b"kepler",
        operands=(xops.ConstantLiteral(c, np.int32(N)), M, ecc),
        shape_with_layout=xla_client.Shape.tuple_shape((M_shape, ecc_shape)),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
            M_shape,
            ecc_shape,
        ),
    )


def kepler_jvp(arg_values, arg_tangents):
    M, e = arg_values
    dM, de = arg_tangents

    sinf, cosf = kepler_p.bind(M, e)

    # Pre-compute some things
    ecosf = e * cosf
    ome2 = 1 - e ** 2

    # Propagate the derivatives
    df = 0.0
    if type(dM) is not ad.Zero:
        df += dM * (1 + ecosf) ** 2 / ome2 ** 1.5
    if type(de) is not ad.Zero:
        df += de * (2 + ecosf) * sinf / ome2

    return (sinf, cosf), (cosf * df, -sinf * df)


def kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return kepler(*args), axes


kepler_p = core.Primitive("kepler")
kepler_p.multiple_results = True
kepler_p.def_impl(partial(xla.apply_primitive, kepler_p))
kepler_p.def_abstract_eval(kepler_abstract_eval)
xla.backend_specific_translations["cpu"][kepler_p] = kepler_translation_rule
ad.primitive_jvps[kepler_p] = kepler_jvp
batching.primitive_batchers[kepler_p] = kepler_batch

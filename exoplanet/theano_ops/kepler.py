# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerOp", "get_eccentric_anomaly", "get_sky_coords"]

import numpy as np

from theano import gof
import theano.tensor as tt


class KeplerOp(gof.Op):

    __props__ = ("tol", "maxiter")

    def __init__(self, tol=1e-8, maxiter=2000, **kwargs):
        self.tol = tol
        self.maxiter = maxiter
        super(KeplerOp, self).__init__(**kwargs)

    def make_node(self, mean_anom, eccen):
        in_args = [tt.as_tensor_variable(mean_anom),
                   tt.as_tensor_variable(eccen)]
        return gof.Apply(self, in_args, [in_args[0].type()])

    def infer_shape(self, node, shapes):
        return shapes[0],

    def c_code_cache_version(self):
        return (0, 0, 1)

    def grad(self, inputs, gradients):
        M, e = inputs
        E = self(M, e)
        bM = gradients[0] / (1.0 - e * tt.cos(E))
        be = tt.sin(E) * bM
        return [bM, be]

    def c_support_code_apply(self, node, name):
        dtype_mean_anom = node.inputs[0].dtype
        dtype_eccen = node.inputs[1].dtype
        dtype_eccen_anom = node.outputs[0].dtype

        c_support_code = """
        inline npy_%(dtype_eccen_anom)s solve_kepler_%(name)s (
            npy_%(dtype_mean_anom)s M, npy_%(dtype_eccen)s e,
            int maxiter, float tol
        ) {
            typedef npy_%(dtype_eccen_anom)s T;
            T E0 = M, E = M;
            if (fabs(e) < tol) return E;
            for (int i = 0; i < maxiter; ++i) {
                T g = E0 - e * sin(E0) - M, gp = 1.0 - e * cos(E0);
                T delta = g / (gp + tol);
                delta = (fabs(delta) < T(1)) ? delta : delta / fabs(delta);
                E = E0 - delta;
                if (fabs(E - E0) <= T(tol)) {
                    return E;
                }
                E0 = E;
            }
            return E;
        }
        """
        return c_support_code % locals()

    def c_code(self, node, name, inp, out, sub):
        tol = self.tol
        maxiter = self.maxiter

        mean_anom, eccen = inp
        eccen_anom, = out

        dtype_mean_anom = node.inputs[0].dtype
        dtype_eccen = node.inputs[1].dtype
        dtype_eccen_anom = node.outputs[0].dtype

        itemsize_mean_anom = np.dtype(dtype_mean_anom).itemsize
        itemsize_eccen = np.dtype(dtype_eccen).itemsize
        itemsize_eccen_anom = np.dtype(dtype_eccen_anom).itemsize

        typenum_eccen_anom = np.dtype(dtype_eccen_anom).num

        fail = sub['fail']

        c_code = """
        npy_intp size = PyArray_SIZE(%(mean_anom)s);

        npy_%(dtype_mean_anom)s* mean_anom;
        npy_%(dtype_eccen)s* eccen;
        npy_%(dtype_eccen_anom)s* eccen_anom;

        // Validate that the inputs have the same shape
        if ( !PyArray_SAMESHAPE(%(mean_anom)s, %(eccen)s) )
        {
            PyErr_Format(PyExc_ValueError, "shape mismatch");
            %(fail)s;
        }

        // Validate that the output storage exists and has the same
        // shape.
        if (NULL == %(eccen_anom)s ||
            !PyArray_SAMESHAPE(%(mean_anom)s, %(eccen_anom)s))
        {
            Py_XDECREF(%(eccen_anom)s);
            %(eccen_anom)s = (PyArrayObject*)PyArray_EMPTY(
                PyArray_NDIM(%(mean_anom)s),
                PyArray_DIMS(%(mean_anom)s),
                %(typenum_eccen_anom)s,
                0);

            if (!%(eccen_anom)s) {
                %(fail)s;
            }
        }

        mean_anom = (npy_%(dtype_mean_anom)s*)PyArray_DATA(%(mean_anom)s);
        eccen = (npy_%(dtype_eccen)s*)PyArray_DATA(%(eccen)s);
        eccen_anom = (npy_%(dtype_eccen_anom)s*)PyArray_DATA(%(eccen_anom)s);

        for (npy_intp i = 0; i < size; ++i) {
            eccen_anom[i] = solve_kepler_%(name)s (mean_anom[i], eccen[i],
                %(maxiter)s, %(tol)s);
        }
        """

        return c_code % locals()


get_eccentric_anomaly = KeplerOp()


def get_sky_coords(period, t0, e, omega, incl, t, **kwargs):
    # Shorthands
    n = 2.0 * np.pi / period
    cos_omega = tt.cos(omega)
    sin_omega = tt.sin(omega)

    # Find the reference time that puts the center of transit at t0
    E0 = 2.0 * tt.arctan2(tt.sqrt(1.0 - e) * cos_omega,
                          tt.sqrt(1.0 + e) * (1.0 + sin_omega))
    tref = t0 - (E0 - e * tt.sin(E0)) / n

    # Solve Kepler's equation for the true anomaly
    M = (t - tref) * n
    kepler = KeplerOp(**kwargs)
    E = kepler(M, e + tt.zeros_like(M))
    f = 2.0 * tt.arctan2(tt.sqrt(1.0 + e) * tt.tan(0.5*E),
                         tt.sqrt(1.0 - e) + tt.zeros_like(E))

    # Rotate into sky coordinates
    cf = tt.cos(f)
    sf = tt.sin(f)
    swpf = sin_omega * cf + cos_omega * sf
    cwpf = cos_omega * cf - sin_omega * sf

    # Star / planet distance
    r = (1.0 - e**2) / (1 + e * cf)

    return tt.stack([
        -r * cwpf,
        -r * swpf * tt.cos(incl),
        r * swpf * tt.sin(incl)
    ])

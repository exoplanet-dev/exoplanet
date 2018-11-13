# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerOp", "get_eccentric_anomaly", "KeplerianOrbit"]

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


# from astropy import constants
# from astropy import units as u

# (constants.M_sun / constants.R_sun**3).to(u.g / u.cm**3).value
gcc_to_sun = 5.905466576479012  # M_sun / R_sun^3 in g / cm^3

# constants.G.to(u.R_sun**3 / u.M_sun / u.day**2).value
G_grav = 2942.206217504419  # R_sun^3 / M_sun / day^2


class KeplerianOrbit(object):

    def __init__(self,
                 period=None, a=None, rho=None,
                 t0=0.0, incl=0.5*np.pi,
                 m_star=None, r_star=None,
                 ecc=None, omega=None,
                 m_planet=0.0, **kwargs):
        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.period = tt.as_tensor_variable(period)
        self.t0 = tt.as_tensor_variable(t0)
        self.incl = tt.as_tensor_variable(incl)
        self.m_planet = tt.as_tensor_variable(m_planet)

        self.a, self.period, self.rho, self.r_star, self.m_star = \
            self._get_consistent_inputs(a, period, rho, r_star, m_star)
        self.m_total = self.m_star + self.m_planet

        self.n = 2 * np.pi / self.period
        self.a_star = self.a * self.m_planet / self.m_total
        self.a_planet = -self.a * self.m_star / self.m_total

        self.K0 = tt.sqrt(G_grav/(self.m_total*self.a))
        self.cos_incl = tt.cos(incl)
        self.sin_incl = tt.sin(incl)

        # Eccentricity
        if ecc is None:
            self.ecc = None
            self.tref = self.t0 - 0.5 * np.pi / self.n
        else:
            self.ecc = tt.as_tensor_variable(ecc)
            if omega is None:
                raise ValueError("both e and omega must be provided")
            self.omega = tt.as_tensor_variable(omega)

            self.cos_omega = tt.cos(self.omega)
            self.sin_omega = tt.sin(self.omega)

            E0 = 2.0 * tt.arctan2(tt.sqrt(1.0-self.ecc)*self.cos_omega,
                                  tt.sqrt(1.0+self.ecc)*(1.0+self.sin_omega))
            self.tref = self.t0 - (E0 - self.ecc * tt.sin(E0)) / self.n

            self.K0 /= tt.sqrt(1 - self.ecc**2)

    def _get_consistent_inputs(self, a, period, rho, r_star, m_star):
        """Check the consistency of the input parameters

        Sample code to work out all the cases using SymPy:

        .. code-block:: python

            import sympy as sm

            rho, a, P, Rs, Ms, G = sm.symbols(
                "rho a period r_star m_star G_grav", positive=True)
            eqs = [
                sm.Eq(rho, Ms / (4 * sm.pi * Rs**3 / 3)),
                sm.Eq((a/Rs)**3, G * rho * P**2 / (3 * sm.pi))
            ]

            syms = [rho, a, P, Rs, Ms]

            for i in range(len(syms)):
                for j in range(i+1, len(syms)):
                    args = (syms[i], syms[j])
                    res = sm.solve(eqs, args)
                    print(args, sm.simplify(res))

        """
        if a is None and period is None:
            raise ValueError("values must be provided for at least one of a "
                             "and period")

        if a is not None:
            a = tt.as_tensor_variable(a)
        if period is not None:
            period = tt.as_tensor_variable(period)

        # Compute the implied density if a and period are given
        if a is not None and period is not None:
            if rho is not None or m_star is not None:
                raise ValueError("if both a and period are given, you can't "
                                 "also define rho or m_star")
            if r_star is None:
                r_star = 1.0
            rho = 3*np.pi*(a / r_star)**3 / (G_grav*period**2)

        # Make sure that the right combination of stellar parameters are given
        if r_star is None and m_star is None:
            r_star = 1.0
            if rho is None:
                m_star = 1.0
        if sum(arg is None for arg in (rho, r_star, m_star)) != 1:
            raise ValueError("values must be provided for exactly two of "
                             "rho, m_star, and r_star")

        if rho is not None:
            # Convert density to M_sun / R_sun^3
            rho = tt.as_tensor_variable(rho) / gcc_to_sun
        if r_star is not None:
            r_star = tt.as_tensor_variable(r_star)
        if m_star is not None:
            m_star = tt.as_tensor_variable(m_star)

        # Work out the stellar parameters
        if rho is None:
            rho = 3*m_star/(4*np.pi*r_star**3)
        elif r_star is None:
            r_star = (3*m_star/(4*np.pi*rho))**(1/3)
        else:
            m_star = 4*np.pi*r_star**3*rho/3

        # Work out the planet parameters
        if a is None:
            a = (G_grav*m_star*period**2/(4*np.pi**2)) ** (1./3)
        elif period is None:
            period = 2*np.pi*a**(3/2)/(np.sqrt(G_grav)*tt.sqrt(m_star))

        return a, period, rho * gcc_to_sun, r_star, m_star

    def _rotate_vector(self, x, y):
        if self.ecc is None:
            a = x
            b = y
        else:
            a = self.cos_omega * x - self.sin_omega * y
            b = self.sin_omega * x + self.cos_omega * y
        return (a, self.cos_incl * b, self.sin_incl * b)

    def _get_true_anomaly(self, t):
        M = (tt.shape_padright(t) - self.tref) * self.n
        if self.ecc is None:
            return M
        E = self.kepler_op(M, self.ecc + tt.zeros_like(M))
        f = 2.0 * tt.arctan2(tt.sqrt(1.0 + self.ecc) * tt.tan(0.5*E),
                             tt.sqrt(1.0 - self.ecc) + tt.zeros_like(E))
        return f

    def _get_position(self, a, t):
        f = self._get_true_anomaly(t)
        cosf = tt.cos(f)
        r = a * (1.0 - self.ecc**2) / (1 + self.ecc * cosf)
        return self._rotate_vector(r * cosf, r * tt.sin(f))

    def get_planet_position(self, t):
        return self._get_position(self.a_planet, t)

    def get_star_position(self, t):
        return self._get_position(self.a_star, t)

    def _get_velocity(self, m, t):
        f = self._get_true_anomaly(t)
        K = self.K0 * m
        return self._rotate_vector(-K*tt.sin(f), K*(tt.cos(f) + self.ecc))

    def get_planet_velocity(self, t):
        return self._get_velocity(-self.m_star, t)

    def get_star_velocity(self, t):
        return self._get_position(self.m_planet, t)

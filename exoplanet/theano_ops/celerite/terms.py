# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Term", "TermSum", "TermProduct", "TermDiff",
    "RealTerm", "ComplexTerm",
    "SHOTerm", "Matern32Term",
]

import numpy as np
from itertools import chain

import theano
import theano.tensor as tt
from theano.ifelse import ifelse


class Term(object):

    parameter_names = tuple()

    def __init__(self, **kwargs):
        self.dtype = kwargs.pop("dtype", theano.config.floatX)
        for name in self.parameter_names:
            if name not in kwargs and "log_" + name not in kwargs:
                raise ValueError(("Missing required parameter {0}. "
                                  "Provide {0} or log_{0}").format(name))
            value = kwargs[name] if name in kwargs \
                else tt.exp(kwargs["log_" + name], name=name)
            setattr(self, name, tt.cast(value, self.dtype))

    def __add__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermSum(self, b, dtype=dtype)

    def __radd__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermSum(b, self, dtype=dtype)

    def __mul__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(self, b, dtype=dtype)

    def __rmul__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(b, self, dtype=dtype)

    def get_real_coefficients(self):
        return (tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype))

    def get_complex_coefficients(self):
        return (tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype))

    def get_coefficients(self):
        r = self.get_real_coefficients()
        c = self.get_complex_coefficients()
        return list(chain(r, c))

    def get_celerite_matrices(self, x, diag):
        x = tt.as_tensor_variable(x)
        diag = tt.as_tensor_variable(diag)
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        a = diag + tt.sum(ar) + tt.sum(ac)
        U = tt.concatenate((
            ar[None, :] + tt.zeros_like(x)[:, None],
            ac[None, :] * tt.cos(dc[None, :] * x[:, None])
            + bc[None, :] * tt.sin(dc[None, :] * x[:, None]),
            ac[None, :] * tt.sin(dc[None, :] * x[:, None])
            - bc[None, :] * tt.cos(dc[None, :] * x[:, None]),
        ), axis=1)

        V = tt.concatenate((
            tt.zeros_like(ar)[None, :] + tt.ones_like(x)[:, None],
            tt.cos(dc[None, :] * x[:, None]),
            tt.sin(dc[None, :] * x[:, None]),
        ), axis=1)

        dx = x[1:] - x[:-1]
        P = tt.concatenate((
            tt.exp(-cr[None, :] * dx[:, None]),
            tt.exp(-cc[None, :] * dx[:, None]),
            tt.exp(-cc[None, :] * dx[:, None]),
        ), axis=1)

        return a, U, V, P

    def to_dense(self, x, diag):
        K = self.value(x[:, None] - x[None, :])
        K += tt.diag(diag)
        return K

    def psd(self, omega):
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        omega = tt.reshape(omega, tt.concatenate([omega.shape, [1]]),
                           ndim=omega.ndim+1)
        w2 = omega**2
        w02 = cc**2 + dc**2
        power = tt.sum(ar * cr / (cr**2 + w2), axis=-1)
        power += tt.sum(((ac*cc+bc*dc)*w02+(ac*cc-bc*dc)*w2) /
                        (w2*w2 + 2.0*(cc**2-dc**2)*w2+w02*w02), axis=-1)
        return np.sqrt(2.0 / np.pi) * power

    def value(self, tau):
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        tau = tt.abs_(tau)
        tau = tt.reshape(tau, tt.concatenate([tau.shape, [1]]),
                         ndim=tau.ndim+1)
        K = tt.sum(ar * tt.exp(-cr*tau), axis=-1)
        factor = tt.exp(-cc*tau)
        K += tt.sum(ac * factor * tt.cos(dc*tau), axis=-1)
        K += tt.sum(bc * factor * tt.sin(dc*tau), axis=-1)
        return K


class TermSum(Term):

    def __init__(self, *terms, **kwargs):
        self.terms = terms
        super(TermSum, self).__init__(**kwargs)

    def get_coefficients(self):
        coeffs = []
        for t in self.terms:
            coeffs.append(t.get_coefficients())
        return [tt.concatenate(a, axis=0) for a in zip(*coeffs)]


class TermProduct(Term):

    def __init__(self, term1, term2, **kwargs):
        self.term1 = term1
        self.term2 = term2
        super(TermProduct, self).__init__(**kwargs)

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        # First compute real terms
        ar = []
        cr = []
        ar.append(tt.flatten(c1[0][:, None] * c2[0][None, :]))
        cr.append(tt.flatten(c1[1][:, None] * c2[1][None, :]))

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        ac.append(tt.flatten(c1[0][:, None] * c2[2][None, :]))
        bc.append(tt.flatten(c1[0][:, None] * c2[3][None, :]))
        cc.append(tt.flatten(c1[1][:, None] + c2[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c1[1])[:, None] + c2[5][None, :]))

        ac.append(tt.flatten(c2[0][:, None] * c1[2][None, :]))
        bc.append(tt.flatten(c2[0][:, None] * c1[3][None, :]))
        cc.append(tt.flatten(c2[1][:, None] + c1[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c2[1])[:, None] + c1[5][None, :]))

        # complex * complex
        aj, bj, cj, dj = c1[2:]
        ak, bk, ck, dk = c2[2:]

        ac.append(tt.flatten(
            0.5*(aj[:, None]*ak[None, :] + bj[:, None]*bk[None, :])))
        bc.append(tt.flatten(
            0.5*(bj[:, None]*ak[None, :] - aj[:, None]*bk[None, :])))
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] - dk[None, :]))

        ac.append(tt.flatten(
            0.5*(aj[:, None]*ak[None, :] - bj[:, None]*bk[None, :])))
        bc.append(tt.flatten(
            0.5*(bj[:, None]*ak[None, :] + aj[:, None]*bk[None, :])))
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] + dk[None, :]))

        return [
            tt.concatenate(vals, axis=0) if len(vals)
            else tt.zeros(0, dtype=self.dtype)
            for vals in (ar, cr, ac, bc, cc, dc)
        ]


class TermDiff(Term):

    def __init__(self, term, **kwargs):
        self.term = term
        super(TermDiff, self).__init__(**kwargs)

    def get_coefficients(self):
        coeffs = self.term.get_coefficients()
        a, b, c, d = coeffs[2:]
        final_coeffs = [
            -coeffs[0]*coeffs[1]**2,
            coeffs[1],
            a*(d**2 - c**2) + 2*b*c*d,
            b*(d**2 - c**2) - 2*a*c*d,
            c, d,
        ]
        return final_coeffs


class RealTerm(Term):

    parameter_names = ("a", "c")

    def get_real_coefficients(self):
        return (
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.c, (self.c.size,)),
        )


class ComplexTerm(Term):

    parameter_names = ("a", "b", "c", "d")

    def get_complex_coefficients(self):
        return (
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.b, (self.b.size,)),
            tt.reshape(self.c, (self.c.size,)),
            tt.reshape(self.d, (self.d.size,)),
        )


class SHOTerm(Term):

    parameter_names = ("S0", "w0", "Q")

    def __init__(self, *args, **kwargs):
        self.eps = tt.as_tensor_variable(kwargs.pop("eps", 1e-5))
        super(SHOTerm, self).__init__(*args, **kwargs)

    def get_coefficients(self):
        def overdampled():
            Q = self.Q
            f = tt.sqrt(tt.maximum(4.0*Q**2 - 1.0, self.eps))
            a = self.S0 * self.w0 * Q
            c = 0.5 * self.w0 / Q
            return (
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
                tt.reshape(a, (a.size,)),
                tt.reshape(a / f, (a.size,)),
                tt.reshape(c, (c.size,)),
                tt.reshape(c * f, (c.size,)),
            )

        def underdamped():
            Q = self.Q
            f = tt.sqrt(tt.maximum(1.0 - 4.0*Q**2, self.eps))
            return (
                0.5*self.S0*self.w0*Q*tt.stack([1.0+1.0/f, 1.0-1.0/f]),
                0.5*self.w0/Q*tt.stack([1.0-f, 1.0+f]),
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
                tt.zeros(0, dtype=self.dtype),
            )

        m = self.Q < 0.5
        return [
            ifelse(m, a, b) for a, b in zip(underdamped(), overdampled())]


class Matern32Term(Term):

    parameter_names = ("sigma", "rho")

    def __init__(self, **kwargs):
        eps = kwargs.pop("eps", None)
        super(Matern32Term, self).__init__(**kwargs)
        if eps is None:
            eps = tt.as_tensor_variable(0.01)
        self.eps = tt.cast(eps, self.dtype)

    def get_complex_coefficients(self):
        w0 = np.sqrt(3.0) / self.rho
        S0 = self.sigma**2 / w0
        return (
            tt.reshape(w0*S0, (w0.size,)),
            tt.reshape(w0*w0*S0/self.eps, (w0.size,)),
            tt.reshape(w0, (w0.size,)),
            tt.reshape(self.eps, (w0.size,))
        )

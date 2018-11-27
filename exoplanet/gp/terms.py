# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Term", "TermSum", "TermProduct", "TermDiff",
    "RealTerm", "ComplexTerm",
    "SHOTerm", "Matern32Term", "RotationTerm",
]

import numpy as np
from itertools import chain

import theano
import theano.tensor as tt
from theano.ifelse import ifelse


class Term(object):
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload the :func:`terms.Term.get_real_coefficients`
    and :func:`terms.Term.get_complex_coefficients` methods.

    """

    parameter_names = tuple()

    def __init__(self, **kwargs):
        self.dtype = kwargs.pop("dtype", theano.config.floatX)
        for name in self.parameter_names:
            if name not in kwargs and "log_" + name not in kwargs:
                raise ValueError(("Missing required parameter {0}. "
                                  "Provide {0} or log_{0}").format(name))
            value = kwargs[name] if name in kwargs \
                else tt.exp(kwargs["log_" + name])
            setattr(self, name, tt.cast(value, self.dtype))

        self.coefficients = self.get_coefficients()

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
        ar, cr, ac, bc, cc, dc = self.coefficients
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
            coeffs.append(t.coefficients)
        return [tt.concatenate(a, axis=0) for a in zip(*coeffs)]


class TermProduct(Term):

    def __init__(self, term1, term2, **kwargs):
        self.term1 = term1
        self.term2 = term2
        super(TermProduct, self).__init__(**kwargs)

    def get_coefficients(self):
        c1 = self.term1.coefficients
        c2 = self.term2.coefficients

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
        coeffs = self.term.coefficients
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
    r"""The simplest celerite term

    This term has the form

    .. math::

        k(\tau) = a_j\,e^{-c_j\,\tau}

    with the parameters ``a`` and ``c``.

    Strictly speaking, for a sum of terms, the parameter ``a`` could be
    allowed to go negative but since it is somewhat subtle to ensure positive
    definiteness, we recommend keeping both parameters strictly positive.
    Advanced users can build a custom term that has negative coefficients but
    care should be taken to ensure positivity.

    Args:
        tensor a or log_a: The amplitude of the term.
        tensor c or log_c: The exponent of the term.

    """

    parameter_names = ("a", "c")

    def get_real_coefficients(self):
        return (
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.c, (self.c.size,)),
        )


class ComplexTerm(Term):
    r"""A general celerite term

    This term has the form

    .. math::

        k(\tau) = \frac{1}{2}\,\left[(a_j + b_j)\,e^{-(c_j+d_j)\,\tau}
         + (a_j - b_j)\,e^{-(c_j-d_j)\,\tau}\right]

    with the parameters ``a``, ``b``, ``c``, and ``d``.

    This term will only correspond to a positive definite kernel (on its own)
    if :math:`a_j\,c_j \ge b_j\,d_j`.

    Args:
        tensor a or log_a: The real part of amplitude.
        tensor b or log_b: The imaginary part of amplitude.
        tensor c or log_c: The real part of the exponent.
        tensor d or log_d: The imaginary part of exponent.

    """

    parameter_names = ("a", "b", "c", "d")

    def get_complex_coefficients(self):
        return (
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.b, (self.b.size,)),
            tt.reshape(self.c, (self.c.size,)),
            tt.reshape(self.d, (self.d.size,)),
        )


class SHOTerm(Term):
    r"""A term representing a stochastically-driven, damped harmonic oscillator

    The PSD of this term is

    .. math::

        S(\omega) = \sqrt{\frac{2}{\pi}} \frac{S_0\,\omega_0^4}
        {(\omega^2-{\omega_0}^2)^2 + {\omega_0}^2\,\omega^2/Q^2}

    with the parameters ``S0``, ``Q``, and ``w0``.

    Args:
        tensor S0 or log_S0: The parameter :math:`S_0`.
        tensor Q or log_Q: The parameter :math:`Q`.
        tensor w0 or log_w0: The parameter :math:`\omega_0`.

    """

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
    r"""A term that approximates a Matern-3/2 function

    This term is defined as

    .. math::

        k(\tau) = \sigma^2\,\left[
            \left(1+1/\epsilon\right)\,e^{-(1-\epsilon)\sqrt{3}\,\tau/\rho}
            \left(1-1/\epsilon\right)\,e^{-(1+\epsilon)\sqrt{3}\,\tau/\rho}
        \right]

    with the parameters ``sigma`` and ``rho``. The parameter ``eps``
    controls the quality of the approximation since, in the limit
    :math:`\epsilon \to 0` this becomes the Matern-3/2 function

    .. math::

        \lim_{\epsilon \to 0} k(\tau) = \sigma^2\,\left(1+
        \frac{\sqrt{3}\,\tau}{\rho}\right)\,
        \exp\left(-\frac{\sqrt{3}\,\tau}{\rho}\right)

    Args:
        tensor sigma or log_sigma: The parameter :math:`\sigma`.
        tensor rho or log_rho: The parameter :math:`\rho`.
        eps (Optional[float]): The value of the parameter :math:`\epsilon`.
            (default: `0.01`)

    """

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


class RotationTerm(TermSum):
    r"""A mixture of two SHO terms that can be used to model stellar rotation

    This term has two modes in Fourier space: one at ``period`` and one at
    ``0.5 * period``. This can be a good descriptive model for a wide range of
    stochastic variability in stellar time series from rotation to pulsations.

    Args:
        tensor amp or log_amp: The amplitude of the variability.
        tensor period or log_period: The primary period of variability.
        tensor Q0 or log_Q0: The quality factor (or really the quality factor
            minus one half) for the secondary oscillation.
        tensor deltaQ or log_deltaQ: The difference between the quality factors
            of the first and the second modes. This parameterization (if
            ``deltaQ > 0``) ensures that the primary mode alway has higher
            quality.
        mix: The fractional amplitude of the secondary mode compared to the
            primary. This should probably always be ``0 < mix < 1``.

    """

    parameter_names = ("amp", "Q0", "deltaQ", "period", "mix")

    def __init__(self, **kwargs):
        super(RotationTerm, self).__init__(**kwargs)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.deltaQ
        w1 = 4*np.pi*Q1/(self.period * tt.sqrt(4*Q1**2-1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8*np.pi*Q2/(self.period * tt.sqrt(4*Q2**2-1))
        S2 = self.mix * self.amp / (w1 * Q2)

        self.terms = (
            SHOTerm(S0=S1, w0=w1, Q=Q1),
            SHOTerm(S0=S2, w0=w2, Q=Q2))
        self.coefficients = self.get_coefficients()

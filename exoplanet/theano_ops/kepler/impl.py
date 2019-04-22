# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_roots_general"]

import numpy as np


def balance_companion_matrix(companion_matrix):
    diag = np.array(np.diag(companion_matrix))
    companion_matrix[np.diag_indices_from(companion_matrix)] = 0.0
    degree = len(diag)

    # gamma <= 1 controls how much a change in the scaling has to
    # lower the 1-norm of the companion matrix to be accepted.
    #
    # gamma = 1 seems to lead to cycles (numerical issues?), so
    # we set it slightly lower.
    gamma = 0.9

    scaling_has_changed = True
    while scaling_has_changed:
        scaling_has_changed = False

        for i in range(degree):
            row_norm = np.sum(np.abs(companion_matrix[i]))
            col_norm = np.sum(np.abs(companion_matrix[:, i]))

            # Decompose row_norm/col_norm into mantissa * 2^exponent,
            # where 0.5 <= mantissa < 1. Discard mantissa (return value
            # of frexp), as only the exponent is needed.
            _, exponent = np.frexp(row_norm / col_norm)
            exponent = exponent // 2

            if exponent != 0:
                scaled_col_norm = np.ldexp(col_norm, exponent)
                scaled_row_norm = np.ldexp(row_norm, -exponent)
                if scaled_col_norm+scaled_row_norm < gamma*(col_norm+row_norm):
                    # Accept the new scaling. (Multiplication by powers of
                    # 2 should not introduce rounding errors (ignoring
                    # non-normalized numbers and over- or underflow))
                    scaling_has_changed = True
                    companion_matrix[i] *= np.ldexp(1.0, -exponent)
                    companion_matrix[:, i] *= np.ldexp(1.0, exponent)

    companion_matrix[np.diag_indices_from(companion_matrix)] = diag
    return companion_matrix


def solve_companion_matrix(poly):
    poly = np.atleast_1d(poly)
    comp = np.eye(len(poly) - 1, k=-1)
    comp[:, -1] = -poly[:-1] / poly[-1]
    return np.linalg.eigvals(balance_companion_matrix(comp))


def get_quadratic(a, e, cosw, sinw, cosi, sini):
    e2 = e*e
    e2mo = e2 - 1
    return (
        (e2*cosw*cosw - 1),
        2*e2*sinw*cosw/sini,
        (e2mo - e2*cosw*cosw)/(sini*sini),
        -2*a*e*e2mo*cosw,
        -2*a*e*e2mo*sinw/sini,
        a**2*e2mo*e2mo,
    )


def get_quartic(A, B, C, D, E, F, T, L):
    A2 = A*A
    B2 = B*B
    C2 = C*C
    D2 = D*D
    E2 = E*E
    F2 = F*F
    T2 = T*T
    L2 = L*L
    return (
        C2*L2*L2 + 2*C*F*L2*T - E2*L2*T + F2*T2,
        -2*T*(B*E*L2 - C*D*L2 - D*F*T),
        2*A*C*L2*T + 2*A*F*T2 - B2*L2*T - 2*C2*L2 - 2*C*F*T + D2*T2 + E2*T,
        2*T*(A*D*T + B*E - C*D),
        A2*T2 - 2*A*C*T + B2*T + C2,
    )


def get_roots_general(a, e, cosw, sinw, cosi, sini, L, tol=1e-8):
    L = L / a
    a = 1.0

    f0 = 2 * np.arctan2(cosw, 1 + sinw)
    default_return = f0 + np.array([-np.pi, np.pi])

    quad = get_quadratic(a, e, cosw, sinw, cosi, sini)
    A, B, C, D, E, F = quad
    T = cosi / sini
    T *= T

    if np.allclose(e, 0.0):
        x2 = (C*L*L + F*T) / (C + T)
        if x2 < 0:
            return default_return, 1
        roots = np.array([np.sqrt(x2), -np.sqrt(x2)])

    elif np.allclose(sinw, 0.0):
        b0 = F*T + C*L*L
        b1 = D*T
        b2 = A*T - C
        x1 = -0.5 * b1 / b2
        arg = b1*b1 - 4*b0*b2
        if arg < 0:
            return default_return, 2
        x2 = 0.5 * np.sqrt(arg) / b2
        roots = np.sort([x1 + x2, x1 - x2])

    else:
        quartic = get_quartic(A, B, C, D, E, F, T, L)

        roots = solve_companion_matrix(quartic)

        # Deal with multiplicity
        roots = roots[np.argsort(np.real(roots))]
        roots = np.real(roots[np.abs(np.imag(roots)) < tol])

    # Fail now if the roots don't span zero
    if np.all(roots <= 0) or np.all(roots > 0):
        return default_return, 3

    # Loop over the roots and compute the z coordinates
    angles = np.empty_like(roots)
    angles[:] = np.nan
    for xi, x in enumerate(roots):
        b0 = A*x*x + D*x + F
        b1 = B*x + E
        b2 = C
        z1 = -0.5 * b1 / b2
        arg = b1*b1 - 4*b0*b2
        if arg < 0:
            continue
        z2 = 0.5 * np.sqrt(arg) / b2
        for sgn in [-1, 1]:
            z = z1 + sgn * z2
            if z > 0:
                continue
            y = z * cosi / sini
            if np.abs(x**2 + y**2 - L**2) > tol:
                continue
            x0 = x*cosw + z*sinw/sini
            y0 = -x*sinw + z*cosw/sini
            angle = np.arctan2(y0, x0) - np.pi
            while angle < -np.pi:
                angle += 2*np.pi
            angles[xi] = angle

    m = np.isfinite(angles)
    if not np.any(m):
        return default_return, 4
    angles = angles[m]
    roots = roots[m]

    # Fail now if the roots don't span zero
    if np.all(roots <= 0) or np.all(roots > 0):
        return default_return, 5

    # Deal with multiplicity
    if len(angles) > 2:
        cosf = np.cos(angles)
        sinf = np.sin(angles)
        e2 = e**2
        dist = (e2 - 1)**2*(cosi**2*(cosw*sinf + sinw*cosf)**2 +
                            (cosw*cosf - sinw*sinf)**2)/(e*cosf + 1)**2 - L**2
        dist = np.abs(dist)

        mm = roots <= 0
        mp = roots > 0
        angles = np.array([
            angles[mm][np.argmin(dist[mm])],
            angles[mp][np.argmin(dist[mp])],
        ])

    # Wrap the roots properly to span the transit
    angles -= f0
    angles = np.sort(angles)
    if len(angles) != 2:
        return default_return, 6

    if np.all(angles > 0):
        angles = np.array([angles[1] - 2*np.pi, angles[0]])
    if np.all(angles < 0):
        angles = np.array([angles[1], angles[0] + 2*np.pi])

    return angles + f0, 0

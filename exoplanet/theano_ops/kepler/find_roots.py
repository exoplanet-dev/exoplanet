# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["find_roots"]

import numpy as np


def find_roots(semimajor, ecc, omega, incl, Ls, tol=1e-6):
    Ls = np.atleast_1d(Ls) / semimajor

    Efactor = np.sqrt((1 - ecc) / (1 + ecc))
    ome2 = 1 - ecc**2
    sinw = np.sin(omega)
    cosw = np.cos(omega)
    sin2w = sinw*sinw
    cos2w = cosw*cosw
    sini = np.sin(incl)
    cosi = np.cos(incl)
    cos2i = cosi * cosi

    f0 = 2 * np.arctan2(cosw, 1 + sinw)

    # a = 1
    # b = 0
    c = 1 / ome2
    d = -2*ecc
    # e = 0
    f = -ome2

    A = cos2i*sin2w + cos2w
    B = 2*cosw*sinw*(cos2i - 1)
    C = cos2i*cos2w + sin2w
    # D = 0
    # E = 0

    results = []
    for L in Ls:
        F = -L*L

        # Warning: special case for constant a, b, e, D, and E
        a0 = C**2*f**2 - 2*C*F*c*f + F**2*c**2
        a1 = 2*C*d*(C*f - F*c)
        a2 = -2*A*C*c*f+2*A*F*c**2+B**2*c*f+C**2*d**2 + 2*C**2*f - 2*C*F*c
        a3 = d*(-2*A*C*c + B**2*c + 2*C**2)
        a4 = A**2*c**2 - 2*A*C*c + B**2*c + C**2

        comp = np.eye(4, k=-1)
        comp[0, -1] = -a0 / a4
        comp[1, -1] = -a1 / a4
        comp[2, -1] = -a2 / a4
        comp[3, -1] = -a3 / a4
        roots = np.linalg.eigvals(comp)

        # Only select real roots
        roots = np.real(roots[np.abs(np.imag(roots)) < tol])

        angles = []
        for x in roots:
            y2 = (1 - (x - ecc)**2) * ome2
            if y2 < 0.0:
                continue

            absy = np.sqrt(y2)
            for sgn in [-1, 1]:
                y = sgn * absy
                x2 = cosw*x - sinw*y
                y2 = cosi*(cosw*y + sinw*x)
                z2 = sini*(cosw*y + sinw*x)

                b2 = x2*x2 + y2*y2
                if np.abs(b2 - L**2) < tol and z2 < 0:
                    angle = np.arctan2(y, x) - np.pi
                    if angle < -np.pi:
                        angle += 2*np.pi
                    angles.append(angle - f0)

        if len(angles) < 2:
            angles = np.array([-np.pi, np.pi])
        else:
            # Deal with multiple roots
            angles = np.array(angles)
            angles.sort()
            delta = np.append(True, np.diff(angles))
            angles = angles[delta > tol]

            # Wrap the roots properly to span the transit
            if len(angles) == 2:
                if np.all(angles > 0):
                    angles = np.array([angles[1] - 2*np.pi, angles[0]])
                if np.all(angles < 0):
                    angles = np.array([angles[1], angles[0] + 2*np.pi])
            else:
                angles = np.array([-np.pi, np.pi])

        # Convert to mean anomaly
        E = 2 * np.arctan(Efactor * np.tan(0.5*(angles + f0)))
        M = E - ecc * np.sin(E)

        results.append(M)

    return results

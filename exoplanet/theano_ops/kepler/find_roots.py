# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GeneralRootFinder", "CircularRootFinder"]

import numpy as np


class GeneralRootFinder(object):

    def __init__(self, semimajor, ecc, omega, incl, tol=1e-6):
        self.ecc = float(ecc)
        self.omega = float(omega)
        self.tol = float(tol)
        self.semimajor = float(semimajor)
        self.incl = float(incl)
        self.sini = np.sin(self.incl)
        self.cosi = np.cos(self.incl)
        self.cos2i = self.cosi * self.cosi
        self.setup_poly()

    def setup_poly(self):
        self.Efactor = np.sqrt((1 - self.ecc) / (1 + self.ecc))
        self.ome2 = 1 - self.ecc**2
        self.sinw = np.sin(self.omega)
        self.cosw = np.cos(self.omega)
        self.sin2w = self.sinw*self.sinw
        self.cos2w = self.cosw*self.cosw
        self.f0 = 2 * np.arctan2(self.cosw, 1 + self.sinw)

        self.a = 1.0
        self.b = 0.0
        self.c = 1 / self.ome2
        self.d = -2*self.ecc
        self.e = 0.0
        self.f = -self.ome2

        self.A = self.cos2i*self.sin2w + self.cos2w
        self.B = 2*self.cosw*self.sinw*(self.cos2i - 1)
        self.C = self.cos2i*self.cos2w + self.sin2w
        self.D = 0.0
        self.E = 0.0

    def setup_quad(self, L):
        F = -(L / self.semimajor)**2

        c, d, f = self.c, self.d, self.f
        A, B, C = self.A, self.B, self.C

        # Warning: special case for constant a, b, e, D, and E
        a0 = C**2*f**2 - 2*C*F*c*f + F**2*c**2
        a1 = 2*C*d*(C*f - F*c)
        a2 = -2*A*C*c*f+2*A*F*c**2+B**2*c*f+C**2*d**2 + 2*C**2*f - 2*C*F*c
        a3 = d*(-2*A*C*c + B**2*c + 2*C**2)
        a4 = A**2*c**2 - 2*A*C*c + B**2*c + C**2

        return a0, a1, a2, a3, a4

    def _get_all_roots(self, L):
        a0, a1, a2, a3, a4 = self. setup_quad(L)

        comp = np.eye(4, k=-1)
        comp[0, -1] = -a0 / a4
        comp[1, -1] = -a1 / a4
        comp[2, -1] = -a2 / a4
        comp[3, -1] = -a3 / a4
        roots = np.linalg.eigvals(comp)

        return roots

    def get_poly(self, L):
        return np.polynomial.Polynomial(self.setup_quad(L))

    def get_real_roots(self, L):
        all_roots = self._get_all_roots(L)

        # Only select real roots
        roots = np.real(all_roots[np.abs(np.imag(all_roots)) < self.tol])

        angles = []
        for x in roots:
            y2 = (1 - (x - self.ecc)**2) * self.ome2
            if y2 < 0.0:
                continue

            absy = np.sqrt(y2)
            for sgn in [-1, 1]:
                y = sgn * absy
                x2 = self.cosw*x - self.sinw*y
                y2 = self.cosi*(self.cosw*y + self.sinw*x)
                z2 = self.sini*(self.cosw*y + self.sinw*x)

                b2 = x2*x2 + y2*y2
                if np.abs(b2 - (L / self.semimajor)**2) < self.tol and z2 < 0:
                    angle = np.arctan2(y, x) - np.pi
                    if angle < -np.pi:
                        angle += 2*np.pi
                    angles.append(angle - self.f0)

        return angles

    def get_unique_and_wrapped_roots(self, L):
        angles = self.get_real_roots(L)

        if len(angles) < 2:
            angles = np.array([-np.pi, np.pi])
        else:
            # Deal with multiple roots
            angles = np.array(angles)
            angles.sort()
            delta = np.append(True, np.diff(angles))
            angles = angles[delta > self.tol]

            # Wrap the roots properly to span the transit
            if len(angles) == 2:
                if np.all(angles > 0):
                    angles = np.array([angles[1] - 2*np.pi, angles[0]])
                if np.all(angles < 0):
                    angles = np.array([angles[1], angles[0] + 2*np.pi])
            else:
                angles = np.array([-np.pi, np.pi])

        return angles

    def get_contact_points(self, L):
        angles = self.get_unique_and_wrapped_roots(L)

        # Convert to mean anomaly
        E = 2 * np.arctan(self.Efactor * np.tan(0.5*(angles + self.f0)))
        M = E - self.ecc * np.sin(E)

        return M


class CircularRootFinder(GeneralRootFinder):

    def __init__(self, semimajor, incl, tol=1e-6):
        super(CircularRootFinder, self).__init__(semimajor, 0.0, 0.0, incl,
                                                 tol=tol)

    def setup_poly(self):
        self.Efactor = 1.0
        self.ome2 = 1.0
        self.sinw = 0.0
        self.cosw = 1.0
        self.sin2w = 0.0
        self.cos2w = 1.0
        self.f0 = 0.5 * np.pi

        self.a = 1.0
        self.b = 0.0
        self.c = 1.0
        self.d = 0.0
        self.e = 0.0
        self.f = -1.0

        self.A = 1.0
        self.B = 0.0
        self.C = self.cosi**2
        self.D = 0.0
        self.E = 0.0

    def setup_quad(self, L):
        F = -(L / self.semimajor)**2
        C = self.C

        # Warning: special case for ecc = 0 and omega = 0
        a0 = C**2 + 2*C*F + F**2
        a1 = 0.0
        a2 = -2*C**2 - 2*C*F + 2*C + 2*F
        a3 = 0.0
        a4 = C**2 - 2*C + 1
        return a0, a1, a2, a3, a4

    def _get_all_roots(self, L):
        a0, a1, a2, a3, a4 = self.setup_quad(L)
        denom = 0.5 / a4
        first = -a2 * denom
        second = np.sqrt(a2 * a2 - 4 * a0 * a4 + 0j) * denom
        roots = np.sqrt([first + second, first - second, 0, 0])
        roots[2:] = -roots[:2]
        return roots

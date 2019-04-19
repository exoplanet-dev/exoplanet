# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GeneralRootFinder", "CircularRootFinder"]

import numpy as np

from .impl import get_roots_general


class GeneralRootFinder(object):

    def __init__(self, semimajor, ecc, omega, incl, tol=1e-6):
        self.tol = float(tol)

        self.ecc = float(ecc)
        self.omega = float(omega)
        self.cosw = np.cos(self.omega)
        self.sinw = np.sin(self.omega)
        self.Efactor = np.sqrt((1 - self.ecc) / (1 + self.ecc))

        self.semimajor = float(semimajor)
        self.incl = float(incl)
        self.cosi = np.cos(self.incl)
        self.sini = np.sin(self.incl)

    def get_contact_points(self, L):
        f = get_roots_general(self.semimajor, self.ecc,
                              self.cosw, self.sinw,
                              self.cosi, self.sini, L, tol=self.tol)

        # Convert to mean anomaly
        E = 2 * np.arctan(self.Efactor * np.tan(0.5*f))
        M = E - self.ecc * np.sin(E)

        return M


class CircularRootFinder(GeneralRootFinder):

    def __init__(self, semimajor, incl, tol=1e-6):
        super(CircularRootFinder, self).__init__(semimajor, 0.0, 0.0, incl,
                                                 tol=tol)

    # def setup_poly(self):
    #     self.Efactor = 1.0
    #     self.ome2 = 1.0
    #     self.sinw = 0.0
    #     self.cosw = 1.0
    #     self.sin2w = 0.0
    #     self.cos2w = 1.0
    #     self.f0 = 0.5 * np.pi

    #     self.a = 1.0
    #     self.b = 0.0
    #     self.c = 1.0
    #     self.d = 0.0
    #     self.e = 0.0
    #     self.f = -1.0

    #     self.A = 1.0
    #     self.B = 0.0
    #     self.C = self.cosi**2
    #     self.D = 0.0
    #     self.E = 0.0

    # def setup_quad(self, L):
    #     F = -(L / self.semimajor)**2
    #     C = self.C

    #     # Warning: special case for ecc = 0 and omega = 0
    #     a0 = C**2 + 2*C*F + F**2
    #     a1 = 0.0
    #     a2 = -2*C**2 - 2*C*F + 2*C + 2*F
    #     a3 = 0.0
    #     a4 = C**2 - 2*C + 1
    #     return a0, a1, a2, a3, a4

    # def _get_all_roots(self, L):
    #     a0, a1, a2, a3, a4 = self.setup_quad(L)
    #     denom = 0.5 / a4
    #     first = -a2 * denom
    #     second = np.sqrt(a2 * a2 - 4 * a0 * a4 + 0j) * denom
    #     roots = np.sqrt([first + second, first - second, 0, 0])
    #     roots[2:] = -roots[:2]
    #     return roots

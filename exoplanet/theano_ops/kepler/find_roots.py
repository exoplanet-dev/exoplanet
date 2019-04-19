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
        f, flag = get_roots_general(self.semimajor, self.ecc,
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

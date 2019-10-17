# -*- coding: utf-8 -*-

__all__ = ["G_grav", "gcc_per_sun", "au_per_R_sun", "day_per_yr_over_2pi"]

import warnings

import astropy.constants as c
import astropy.units as u
import numpy as np

try:
    G_grav = c.G.to(u.R_sun ** 3 / u.M_sun / u.day ** 2).value
    gcc_per_sun = (u.M_sun / u.R_sun ** 3).to(u.g / u.cm ** 3)
    au_per_R_sun = u.R_sun.to(u.au)

    day_per_yr_over_2pi = (
        (1.0 * u.au) ** (3 / 2)
        / (
            np.sqrt(
                c.G.to(u.au ** 3 / (u.M_sun * u.day ** 2)) * (1.0 * u.M_sun)
            )
        )
    ).value

except TypeError:
    warnings.warn(
        "Something went wrong when computing units. "
        "Check that AstroPy is working.",
        category=RuntimeWarning,
    )
    G_grav = 2942.2062175044193
    gcc_per_sun = 5.905271918964842
    au_per_R_sun = 0.00465046726096215
    day_per_yr_over_2pi = 58.13244087623439

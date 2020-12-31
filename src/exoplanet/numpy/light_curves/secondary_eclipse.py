# -*- coding: utf-8 -*-

__all__ = ["SecondaryEclipseLightCurve"]

from .. import compat
from .limb_dark import LimbDarkLightCurve


class SecondaryEclipseLightCurve:
    """The light curve for a secondary eclipse model computed using starry

    Args:
        u_primary (tuple): The limb darkening coefficients for the
            primary body.
        u_secondary (tuple): The limb darkening coefficients for the
            secondary body.
        surface_brightness_ratio (scalar): The surface brightness ratio of the
            secondary with respect to the primary.

    """

    def __init__(
        self, u_primary, u_secondary, surface_brightness_ratio, *, model=None
    ):
        self.primary = LimbDarkLightCurve(*u_primary, model=model)
        self.secondary = LimbDarkLightCurve(*u_secondary, model=model)
        self.surface_brightness_ratio = compat.as_tensor(
            surface_brightness_ratio
        )

    def get_light_curve(
        self,
        *,
        orbit,
        t,
        texp=None,
        oversample=7,
        order=0,
        use_in_transit=None,
        light_delay=False,
    ):
        orbit2 = orbit._flip()
        lc1 = self.primary.get_light_curve(
            orbit=orbit,
            t=t,
            texp=texp,
            oversample=oversample,
            order=order,
            use_in_transit=use_in_transit,
            light_delay=light_delay,
        )
        lc2 = self.secondary.get_light_curve(
            orbit=orbit2,
            t=t,
            texp=texp,
            oversample=oversample,
            order=order,
            use_in_transit=use_in_transit,
            light_delay=light_delay,
        )

        k = orbit.r_planet / orbit.r_star
        flux_ratio = self.surface_brightness_ratio * k ** 2

        return (lc1 + flux_ratio * lc2) / (1 + flux_ratio)

__all__ = [
    "ImpactParameter",
    "QuadLimbDark",
    "angle",
    "unit_disk",
    "impact_parameter",
    "quad_limb_dark",
    "eccentricity",
]

from exoplanet.distributions import eccentricity
from exoplanet.distributions.distributions import (
    angle,
    unit_disk,
    impact_parameter,
    quad_limb_dark,
)

# For backwards compatibility, define wrappers for the old Distribution-based
# versions of these functions
from exoplanet.utils import deprecated

ImpactParameter = deprecated(
    alternate="exoplanet.distributions.impact_parameter"
)(impact_parameter)
QuadLimbDark = deprecated(alternate="exoplanet.distributions.quad_limb_dark")(
    quad_limb_dark
)

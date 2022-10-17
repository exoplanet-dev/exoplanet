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
    impact_parameter,
    quad_limb_dark,
    unit_disk,
)
from exoplanet.utils import deprecated

# For backwards compatibility, define wrappers for the old Distribution-based
# versions of these functions
ImpactParameter = deprecated(
    alternate="exoplanet.distributions.impact_parameter"
)(impact_parameter)
QuadLimbDark = deprecated(alternate="exoplanet.distributions.quad_limb_dark")(
    quad_limb_dark
)

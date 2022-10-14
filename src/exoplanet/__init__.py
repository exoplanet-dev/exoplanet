from exoplanet import distributions as distributions
from exoplanet import interp as interp
from exoplanet import orbits as orbits
from exoplanet.citations import CITATIONS as CITATIONS
from exoplanet.distributions import *  # NOQA
from exoplanet.estimators import *  # NOQA
from exoplanet.exoplanet_version import __version__ as __version__
from exoplanet.light_curves import *  # NOQA
from exoplanet.utils import *  # NOQA

__bibtex__ = __citation__ = CITATIONS["exoplanet"][1]
__uri__ = "https://docs.exoplanet.codes"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "Fast and scalable MCMC for all your exoplanet needs"

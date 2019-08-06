get_ipython().magic("matplotlib inline")
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 20
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
rcParams["figure.autolayout"] = True

# Hide deprecation warnings from Theano
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Hide Theano compilelock warnings
import logging

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)

import theano

print("theano version: {0}".format(theano.__version__))

import pymc3

print("pymc3 version: {0}".format(pymc3.__version__))

import exoplanet

print("exoplanet version: {0}".format(exoplanet.__version__))

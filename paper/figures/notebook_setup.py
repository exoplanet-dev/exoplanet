get_ipython().magic('config InlineBackend.figure_format = "retina"')

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

import matplotlib.pyplot as plt
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
# plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ("Liberation Serif", "DejaVu Serif")
plt.rcParams["mathtext.fontset"] = "custom"
# plt.rcParams["mathtext.rm"] = "Liberation Serif"
# plt.rcParams["mathtext.it"] = "Liberation Serif"
# plt.rcParams["mathtext.bf"] = "Liberation Serif"
# plt.rcParams["text.usetex"] = False

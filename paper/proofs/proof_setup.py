get_ipython().magic("matplotlib inline")
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 16
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

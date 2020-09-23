# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run notebook_setup

# %% [markdown]
# # Citing exoplanet & its dependencies

# %% [markdown]
# The *exoplanet* package is mostly just glue that connects many other ideas and software.
# In a situation like this, it can be easy to forget about the important infrastructure upon which our science is built.
# In order to make sure that you can easily give credit where credit is due, we have tried to make it as painless as possible to work out which citations are expected for a model fit using *exoplanet* by including a :func:`exoplanet.citations.get_citations_for_model` function that introspects the current PyMC3 model and constructs a list of citations for the functions used in that model.
#
# For example, you might compute a quadratically limb darkened light curve using `starry` (via the :class:`exoplanet.LimbDarkLightCurve` class):

# %%
import pymc3 as pm
import exoplanet as xo

with pm.Model() as model:
    u = xo.distributions.QuadLimbDark("u")
    orbit = xo.orbits.KeplerianOrbit(period=10.0)
    light_curve = xo.LimbDarkLightCurve(u)
    transit = light_curve.get_light_curve(r=0.1, orbit=orbit, t=[0.0, 0.1])

    txt, bib = xo.citations.get_citations_for_model()

# %% [markdown]
# The :func:`exoplanet.citations.get_citations_for_model` function would generate an acknowledgement that cites:
#
# * [PyMC3](https://docs.pymc.io/#citing-pymc3): for the inference engine and modeling framework,
# * [Theano](http://deeplearning.net/software/theano/citation.html): for the numerical infrastructure,
# * [AstroPy](http://www.astropy.org/acknowledging.html): for units and constants,
# * [Kipping (2013)](https://arxiv.org/abs/1308.0009): for the reparameterization of the limb darkening parameters for a quadratic law, and
# * [Luger, et al. (2018)](https://arxiv.org/abs/1810.06559): for the light curve calculation.
#
# The first output from :func:`exoplanet.citations.get_citations_for_model` gives the acknowledgement text:

# %%
print(txt)

# %% [markdown]
# And the second output is a string with BibTeX entries for each of the citations in the acknowledgement text:

# %%
print(bib)

# %%

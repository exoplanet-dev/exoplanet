---
title: "`exoplanet`: Gradient-based probabilistic inference for exoplanet data & other astronomical time series"
tags:
  - Python
  - astronomy
authors:
  - name: Daniel Foreman-Mackey^[Author order TBD]
    orcid: 0000-0002-9328-5652
    affiliation: 1
  - name: Thomas Barclay
    orcid: 0000-0001-7139-2724
    affiliation: 2
  - name: Jiayin Dong
    orcid: 0000-0002-3610-6953
    affiliation: "3, 4"
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, USA
    index: 1
  - name: University of Maryland Baltimore County, Baltimore, MD, USA
    index: 2
  - name: The Pennsylvania State University, University Park, PA, USA
    index: 3
  - name: Center for Exoplanets & Habitable Worlds, The Pennsylvania State University, University Park, PA, USA
    index: 4
date: 23 April 2021
bibliography: paper.bib
---

# Summary

`exoplanet` is a toolkit for probabilistic modeling of time series data in
astronomy with a focus on observations of exoplanets, using `PyMC3` [@pymc3].
`PyMC3` is a flexible and high-performance model building language and inference
engine that scales well to problems with a large number of parameters.
`exoplanet` extends `PyMC3`’s modeling language to support many of the custom
functions and distributions required when fitting exoplanet datasets or other
exoplanet time series.

While it has been used for other applications, such as the study of stellar
rotation [@gillen20; @medina20], the primary purpose of `exoplanet` is the
characterization of exoplanets or binary star systems using transit light
curves, radial velocity time series, and/or astrometric time series. In
particular, the typical use case would be to use one or more of these datasets
to place constraints on the physical and orbital parameters of the system, such
as planet mass or orbital period, for example.

# Statement of need

Time domain astronomy is a priority of the observational astronomical community,
with huge survey datasets currently available and more forthcoming. Within this
research domain, there is significant investment into the discovery and
characterization of exoplanets, planets orbiting stars other than our Sun. These
datasets are large (on the scale of hundreds of thousands of observations per
star from space-based observatories such as _Kepler_ and _TESS_) and the
research questions are becoming more ambitious (in terms of both the
computational cost of the physical models, and the flexibility of these models).
The packages in the _exoplanet_ ecosystem are designed to enable rigorous
probabilistic inference with these large datasets and high-dimensional models by
providing a high-performance and well-tested infrastructure for integrating
these models with modern modeling frameworks such as `PyMC3`. Since its initial
release at the end of 2018, `exoplanet` has been widely used with HOWMANY
citations of the Zenodo record [@zenodo].

# The _exoplanet_ software ecosystem

Besides the primary `exoplanet` package, the _exoplanet_ ecosystem of projects
includes several other libraries. This paper describes and is the primary
reference for this full suite of packages. The following provides a short
description of each library within this ecosystem and discusses how they are
related.

- `exoplanet`[^exoplanet] is the primary library and it includes implementations
  of many special functions required for exoplanet data analysis. This includes
  the spherical geometry for computing orbits, some exoplanet-specific
  distributions for eccentricity [@kipping13b; @vaneylen19] and limb darkening
  [@kipping13], and exposure-time integrated limb darkened transit light curves.
- `exoplanet-core`[^exoplanet-core] provides efficient, well-tested, and
  differentiable implementations of all of the exoplanet-specific operations
  that must be compiled for performance. This includes an efficient solver for
  Kepler's equation and limb darkened transit light curves [agol20]. Besides the
  implementation for `PyMC3` and `Theano`, `exoplanet-core` includes
  implementations in `numpy` and `jax`.
- `celerite2`[^celerite2], an updated implementation of the _celerite_
  algorithm[^celerite] [@foremanmackey17; @foremanmackey18] for scalable
  Gaussian Process regression for time series data. Like `exoplanet-core`,
  `celerite2` includes support for `numpy`, `jax`, and `PyMC3`, as well as some
  recent generalizations of the _celerite_ algorithm [@gordon20].
- `pymc3-ext`[^pymc3-ext], a set of helper functions to make `PyMC3` more
  amenable to the standard astronomical workflow. For example, it provides a
  tuning schedule for `PyMC3`'s sampler [based on the method used by the `Stan`
  project @carpenter17] that provides better performance on models with
  correlated parameters.
- `rebound-pymc3`[^rebound-pymc3] provides an interface between _REBOUND_
  [@rein12], _REBOUNDx_ [@tamayo20], and `PyMC3` to enable inference with full
  N-body orbit integration.

# Documentation & case studies

The main documentation page for the _exoplanet_ libraries lives at
[docs.exoplanet.codes](https://docs.exoplanet.codes) where it is hosted on
[ReadTheDocs](https://readthedocs.org). The tutorials included with the
documentation are automatically executed on every push or pull request to the
GitHub repository, with the goal of ensuring that the tutorials are always
compatible with the current version of the code. The `celerite2` project has its
own documentation page at
[celerite2.readthedocs.io](https://celerite2.readthedocs.io), with tutorials
that are similarly automatically executed.

Alongside these documentation pages, there is a parallel "Case Studies" website
at [gallery.exoplanet.codes](https://gallery.exoplanet.codes) that includes more
detailed example use cases for `exoplanet` and the other libraries described
here. Like the tutorials on the documentation page, these case studies are
automatically executed using GitHub Actions, but at a less regular cadence (once
a week and when are new release of the `exoplanet` library is made) since the
runtime is much longer.

# Similar tools

There is a rich ecosystem of tooling available for inference with models such as
the ones supported by `exoplanet`. Each of these tools has their own set of
strengths and limitations and we will not make a detailed comparison here, but
it is worth listing some of these tools and situating `exoplanet` in this
context.

Some of the most popular tools in this space include (and note that this is far
from a comprehensive list!) `EXOFAST` [@eastman13; @eastman19], `radvel`
[@fulton18], `juliet` [@espinoza19], `exostriker` [@trifonov19], `PYANETI`
[@barragan19], `allesfitter` [@guenther20], and `orbitize` [@blunt20]. These
packages all focus on providing a high level interface for designing models and
then executing a fit. `exoplanet`, however, is designed to be lower level, and
more conceptually similar to tools like `batman` [@kreidberg15], `PyTransit`
[@parviainen15], `ellc` [@maxted16], or `starry` [@luger19], which provide the
building blocks for evaluating the models required for inference with exoplanet
datasets. In fact, several of the higher-level packages listed above include
these lower-level libraries as dependencies and our hope is that `exoplanet`
could provide the backend for future high-level libraries.

As emphasized in the title of this paper, the main selling point `exoplanet`
when compared to other tools in this space is that it supports differentiation
of all components of the model and is designed to integrate seamlessly with the
`aesara` [@aesara; formerly known as `Theano`, @theano] automatic differentiation
framework used by `PyMC3`. This allows the use of modern inference algorithms
such as No U-Turn Sampling [@hoffman14] or Automatic Differentiation Variational
Inference [@kucukelbir17].

# Acknowledgements

We would like to thank the Astronomical Data Group at Flatiron for listening to
every iteration of this project and for providing great feedback every step of
the way.

Besides the software cited above, `exoplanet` is also built on top of `ArviZ`
[@arviz], `AstroPy` [@astropy13; @astropy18], and `NumPy` [@numpy].

# References

[^exoplanet]: [https://github.com/exoplanet-dev/exoplanet](https://github.com/exoplanet-dev/exoplanet)
[^exoplanet-core]: [https://github.com/exoplanet-dev/exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core)
[^celerite2]: [https://celerite2.readthedocs.io](https://celerite2.readthedocs.io)
[^celerite]: [https://celerite.readthedocs.io](https://celerite.readthedocs.io)
[^pymc3-ext]: [https://github.com/exoplanet-dev/pymc3-ext](https://github.com/exoplanet-dev/pymc3-ext)
[^rebound-pymc3]: [https://github.com/exoplanet-dev/rebound-pymc3](https://github.com/exoplanet-dev/rebound-pymc3)

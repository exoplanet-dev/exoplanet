---
title: "`exoplanet`: Gradient-based probabilistic inference for exoplanet data & other astronomical time series"
tags:
  - Python
  - astronomy
authors:
  - name: Daniel Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: 1
  - name: Rodrigo Luger
    orcid: 0000-0002-0296-3826
    affiliation: "1,2"
  - name: Eric Agol
    orcid: 0000-0002-0802-9145
    affiliation: "3,2"
  - name: Thomas Barclay
    orcid: 0000-0001-7139-2724
    affiliation: 4
  - name: Luke G. Bouma
    orcid: 0000-0002-0514-5538
    affiliation: 5
  - name: Timothy D. Brandt
    orcid: 0000-0003-2630-8073
    affiliation: 6
  - name: Ian Czekala
    orcid: 0000-0002-1483-8811
    affiliation: "7,8,9,10"
  - name: Trevor J. David
    orcid: 0000-0001-6534-6246
    affiliation: "1,11"
  - name: Jiayin Dong
    orcid: 0000-0002-3610-6953
    affiliation: "7,8"
  - name: Emily A. Gilbert
    orcid: 0000-0002-0388-8004
    affiliation: 12
  - name: Tyler A. Gordon
    orcid: 0000-0001-5253-1987
    affiliation: 3
  - name: Christina Hedges
    orcid: 0000-0002-3385-8391
    affiliation: "13,14"
  - name: Daniel R. Hey
    orcid: 0000-0003-3244-5357
    affiliation: "15,16"
  - name: Brett M. Morris
    orcid: 0000-0003-2528-3409
    affiliation: 17
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Arjun B. Savel
    orcid: 0000-0002-2454-768X
    affiliation: 18
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, USA
    index: 1
  - name: Virtual Planetary Laboratory, University of Washington, Seattle, WA, USA
    index: 2
  - name: Department of Astronomy, University of Washington, University of Washington, Seattle, WA, USA
    index: 3
  - name: Center for Space Sciences and Technology, University of Maryland, Baltimore County, Baltimore, MD, USA
    index: 4
  - name: Department of Astrophysical Sciences, Princeton University, Princeton, NJ, USA
    index: 5
  - name: Department of Physics, University of California, Santa Barbara, Santa Barbara, CA, USA
    index: 6
  - name: Department of Astronomy and Astrophysics, The Pennsylvania State University, University Park, PA, USA
    index: 7
  - name: Center for Exoplanets and Habitable Worlds, The Pennsylvania State University, University Park, PA, USA
    index: 8
  - name: Center for Astrostatistics, The Pennsylvania State University, University Park, PA, USA
    index: 9
  - name: Institute for Computational and Data Sciences, The Pennsylvania State University, University Park, PA, USA
    index: 10
  - name: Department of Astrophysics, American Museum of Natural History, New York, NY, USA
    index: 11
  - name: Department of Astronomy and Astrophysics, University of Chicago, Chicago, IL, USA
    index: 12
  - name: NASA Ames Research Center, Moffett Field, CA, USA
    index: 13
  - name: Bay Area Environmental Research Institute, Moffett Field, CA, USA
    index: 14
  - name: Sydney Institute for Astronomy, School of Physics, University of Sydney, Camperdown, New South Wales, Australia
    index: 15
  - name: Stellar Astrophysics Centre, Department of Physics and Astronomy, Aarhus University, Aarhus, Denmark
    index: 16
  - name: Center for Space and Habitability, University of Bern, Bern, Switzerland
    index: 17
  - name: Department of Astronomy, University of Maryland, College Park, MD, USA
    index: 18
date: 23 April 2021
bibliography: paper.bib
---

# Summary

`exoplanet` is a toolkit for probabilistic modeling of astronomical time series
data, with a focus on observations of exoplanets, using `PyMC3` [@pymc3].
`PyMC3` is a flexible and high-performance model building language and inference
engine that scales well to problems with a large number of parameters.
`exoplanet` extends `PyMC3`’s modeling language to support many of the custom
functions and probability distributions required when fitting exoplanet datasets
or other astronomical time series.

While it has been used for other applications, such as the study of stellar
variability [e.g., @gillen20; @medina20], the primary purpose of `exoplanet` is
the characterization of exoplanets [e.g., @gilbert20; @plavchan20] or multiple
star systems [e.g., @czekala21] using time series photometry, astrometry, and/or
radial velocity. In particular, the typical use case would be to use one or more
of these datasets to place constraints on the physical and orbital parameters of
the system, such as planet mass or orbital period, while simultaneously taking
into account the effects of stellar variability.

# Statement of need

Time domain astronomy is a priority of the observational astronomical community,
with huge survey datasets currently available and more forthcoming. Within this
research domain, there is significant investment into the discovery and
characterization of exoplanets, planets orbiting stars other than our Sun. These
datasets are large (on the scale of hundreds of thousands of observations per
star from space-based observatories such as _Kepler_ and _TESS_), and the
research questions are becoming more ambitious (in terms of both the
computational cost of the physical models and the flexibility of these models).
The packages in the _exoplanet_ ecosystem are designed to enable rigorous
probabilistic inference with these large datasets and high-dimensional models by
providing a high-performance and well-tested infrastructure for integrating
these models with modern modeling frameworks such as `PyMC3`. Since its initial
release at the end of 2018, `exoplanet` has been widely used, with 64
citations of the Zenodo record [@zenodo] so far.

# The _exoplanet_ software ecosystem

Besides the primary `exoplanet` package, the _exoplanet_ ecosystem of projects
includes several other libraries. This paper describes, and is the primary
reference for, this full suite of packages. The following provides a short
description of each library within this ecosystem and discusses how they are
related.

- `exoplanet`[^exoplanet] is the primary library, and it includes implementations
  of many special functions required for exoplanet data analysis. These include
  the spherical geometry for computing orbits, some exoplanet-specific
  distributions for eccentricity [@kipping13b; @vaneylen19] and limb darkening
  [@kipping13], and exposure-time integrated limb darkened transit light curves.
- `exoplanet-core`[^exoplanet-core] provides efficient, well-tested, and
  differentiable implementations of all of the exoplanet-specific operations
  that must be compiled for performance. These include an efficient solver for
  Kepler's equation [based on the algorithm proposed by @raposo17] and limb
  darkened transit light curves [@agol20]. Besides the implementation for
  `PyMC3`, `exoplanet-core` includes implementations in `numpy` [@numpy] and
  `jax` [@jax].
- `celerite2`[^celerite2], is an updated implementation of the _celerite_
  algorithm[^celerite] [@foremanmackey17; @foremanmackey18] for scalable
  Gaussian Process regression for time series data. Like `exoplanet-core`,
  `celerite2` includes support for `numpy`, `jax`, and `PyMC3`, as well as some
  recent generalizations of the _celerite_ algorithm [@gordon20].
- `pymc3-ext`[^pymc3-ext], includes a set of helper functions to make `PyMC3`
  more amenable to the typical astronomical data analysis workflow. For example,
  it provides a tuning schedule for `PyMC3`'s sampler [based on the method used
  by the `Stan` project and described by @carpenter17] that provides better
  performance on models with correlated parameters.
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
automatically executed using GitHub Actions, but at lower cadence (once a week
and when a new release of the `exoplanet` library is made) since the runtime is
much longer. \autoref{fig:figure} shows the results of two example case studies
demonstrating some of the potential use cases of the `exoplanet` software
ecosystem.

![Some examples of datasets fit using `exoplanet`. The full analyses behind
these examples are available on the "Case Studies" page as Jupyter notebooks.
(left) A fit to the light curves of a transiting exoplanet observed by two
different space-based photometric surveys: Kepler and TESS. (right) The phase
folded radial velocity time series for an exoplanet observed from different
observatories with different instruments, fit simultaneously using `exoplanet`.
\label{fig:figure}](figures/figure.png)

# Similar tools

There is a rich ecosystem of tooling available for inference with models such as
the ones supported by `exoplanet`. Each of these tools has its own set of
strengths and limitations and we will not make a detailed comparison here, but
it is worth listing some of these tools and situating `exoplanet` in this
context.

Some of the most popular tools in this space include (and note that this is far
from a comprehensive list!) `EXOFAST` [@eastman13; @eastman19], `radvel`
[@fulton18], `juliet` [@espinoza19], `exostriker` [@trifonov19], `PYANETI`
[@barragan19], `allesfitter` [@guenther20], and `orbitize` [@blunt20]. Similar
tools also exist for modeling observations of eclipsing binary systems,
including `JKTEBOP` [@southworth04], `eb` [@irwin11], and `PHOEBE` [@conroy20].
These packages all focus on providing a high-level interface for designing
models and then executing a fit. `exoplanet`, however, is designed to be lower
level and more conceptually similar to tools like `batman` [@kreidberg15],
`PyTransit` [@parviainen15], `ldtk` [@parviainen15b], `ellc` [@maxted16],
`starry` [@luger19], or `Limbdark.jl` [@agol20], which provide the building
blocks for evaluating the models required for inference with exoplanet datasets.
In fact, several of the higher-level packages listed above include these
lower-level libraries as dependencies, and our hope is that `exoplanet` could
provide the backend for future high-level libraries.

As emphasized in the title of this paper, the main selling point of `exoplanet`
when compared to other tools in this space is that it supports differentiation
of all components of the model and is designed to integrate seamlessly with the
`aesara` [@aesara] automatic differentiation framework used by `PyMC3`. It is
worth noting that `aesara` was previously known as `Theano` [@theano], so these
names are sometimes used interchangeably in the `PyMC3` or `exoplanet`
documentation[^theano-aesara]. This allows the use of modern inference
algorithms such as No U-Turn Sampling [@hoffman14] or Automatic Differentiation
Variational Inference [@kucukelbir17]. These algorithms can have some
computational and conceptual advantages over inference methods that do not use
gradients, especially for high-dimensional models. The computation of gradients
is also useful for model optimization; this is necessary when, say, searching
for new exoplanets, mapping out degeneracies or multiple modes of a posterior,
or estimating uncertainties from a Hessian. Care has been taken to provide
gradients which are numerically stable, and more accurate and faster to evaluate
than finite-difference gradients.

# Acknowledgements

We would like to thank the Astronomical Data Group at Flatiron for listening to
every iteration of this project and for providing great feedback every step of
the way.

This research was partially conducted during the _Exostar19_ program at the
_Kavli Institute for Theoretical Physics_ at UC Santa Barbara, which was
supported in part by the National Science Foundation under Grant No. NSF
PHY-1748958.

Besides the software cited above, `exoplanet` is also built on top of `ArviZ`
[@arviz] and `AstroPy` [@astropy13; @astropy18].

# References

[^exoplanet]: [https://github.com/exoplanet-dev/exoplanet](https://github.com/exoplanet-dev/exoplanet)
[^exoplanet-core]: [https://github.com/exoplanet-dev/exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core)
[^celerite2]: [https://celerite2.readthedocs.io](https://celerite2.readthedocs.io)
[^celerite]: [https://celerite.readthedocs.io](https://celerite.readthedocs.io)
[^pymc3-ext]: [https://github.com/exoplanet-dev/pymc3-ext](https://github.com/exoplanet-dev/pymc3-ext)
[^rebound-pymc3]: [https://github.com/exoplanet-dev/rebound-pymc3](https://github.com/exoplanet-dev/rebound-pymc3)
[^theano-aesara]: More information about this distinction is available at [https://docs.exoplanet.codes/en/stable/user/theano/](https://docs.exoplanet.codes/en/stable/user/theano/)

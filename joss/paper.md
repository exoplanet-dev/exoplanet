---
title: "`exoplanet`: Gradient-based probabilistic inference for exoplanet data & other astronomical time series"
tags:
  - Python
  - astronomy
authors:
  - name: Daniel Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: 1
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY
    index: 1
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
exoplanet time series. Besides the primary `exoplanet` package, the _exoplanet_
ecosystem of projects includes:

- `exoplanet-core`[^exoplanet-core], which includes efficient and well-tested
  compiled implementations of all of the exoplanet-specific operations required
  by the exoplanet package;
- `celerite2`[^celerite2], an updated implementation of the _celerite_
  algorithm[^celerite] [@foremanmackey17, @foremanmackey18] for scalable
  Gaussian Process regression for time series data;
- `pymc3-ext`[^pymc3-ext], a set of helper functions to make `PyMC3` more
  amenable to the standard astronomical workflow; and
- `rebound-pymc3`[^rebound-pymc3], an interface between _REBOUND_ [@rein12],
  _REBOUNDx_ [@tamayo20], and `PyMC3` to enable inference with full N-body orbit
  integration.

This paper describes this suite of packages and their relationships.

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

## exoplanet

## exoplanet-core

## celerite2

## pymc3-ext

## rebound-pymc3

# Similar tools

There is a rich ecosystem of tooling available for inference with models such as
the ones supported by `exoplanet`. Each of these tools has their own set of
strengths and limitations and we will not make a detailed comparison here, but
it is worth listing some of these tools and situating `exoplanet` in this
context.

Some of the most popular tools in this space include `EXOFAST` [@eastman13,
@eastman19], `radvel` [@fulton18], `juliet` [@espinoza19], `allesfitter`
[@guenther20], and `orbitize` [@blunt20]. These packages all focus on providing
a high level interface for designing models and then executing a fit.
`exoplanet`, however, is designed to be lower level, and more conceptually
similar to tools like `batman` [@kreidberg15], `ellc` [@maxted16], or `starry`
[@luger19], which provide the building blocks for evaluating the models required
for inference with exoplanet datasets. In fact, several of the higher-level
packages listed above include these lower-level libraries as dependencies and
our hope is that `exoplanet` could provide the backend for future high-level
libraries.

As emphasized in the title of this paper, the main selling point `exoplanet`
when compared to other tools in this space is that it supports differentiation
of all components of the model and is designed to integrate seamlessly with the
`aesara` [formerly known as `Theano` @theano] automatic differentiation
framework used by `PyMC3`. This allows the use of modern inference algorithms
such as No U-Turn Sampling [@hoffman14] or Automatic Differentiation Variational
Inference [@kucukelbir17].

# References

[^exoplanet-core]: [https://github.com/exoplanet-dev/exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core)
[^celerite2]: [https://celerite2.readthedocs.io](https://celerite2.readthedocs.io)
[^celerite]: [https://celerite.readthedocs.io](https://celerite.readthedocs.io)
[^pymc3-ext]: [https://github.com/exoplanet-dev/pymc3-ext](https://github.com/exoplanet-dev/pymc3-ext)
[^rebound-pymc3]: [https://github.com/exoplanet-dev/rebound-pymc3](https://github.com/exoplanet-dev/rebound-pymc3)

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

_exoplanet_ is a toolkit for probabilistic modeling of time series data in
astronomy with a focus on observations of exoplanets, using _PyMC3_ [@pymc3].
PyMC3 is a flexible and high-performance model building language and inference
engine that scales well to problems with a large number of parameters.
_exoplanet_ extends _PyMC3_’s modeling language to support many of the custom
functions and distributions required when fitting exoplanet datasets or other
exoplanet time series. Besides the primary `exoplanet` package, the _exoplanet_
ecosystem of projects includes:

- `exoplanet-core`[^exoplanet-core], which includes efficient and well-tested
  compiled implementations of all of the exoplanet-specific operations required
  by the exoplanet package;
- `celerite2`[^celerite2], an updated implementation of the _celerite_
  algorithm[^celerite] [@foremanmackey17, @foremanmackey18] for scalable
  Gaussian Process regression for time series data;
- `pymc3-ext`[^pymc3-ext], a set of helper functions to make _PyMC3_ more
  amenable to the standard astronomical workflow; and
- `rebound-pymc3`[^rebound-pymc3], an interface between _rebound_ [@rein12] and
  _PyMC3_ to enable inference with full N-body orbit integration.

# Statement of need

Some words

# References

[^exoplanet-core]: [https://github.com/exoplanet-dev/exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core)
[^celerite2]: [https://celerite2.readthedocs.io](https://celerite2.readthedocs.io)
[^celerite]: [https://celerite.readthedocs.io](https://celerite.readthedocs.io)
[^pymc3-ext]: [https://github.com/exoplanet-dev/pymc3-ext](https://github.com/exoplanet-dev/pymc3-ext)
[^rebound-pymc3]: [https://github.com/exoplanet-dev/rebound-pymc3](https://github.com/exoplanet-dev/rebound-pymc3)

exoplanet
=========

<p>
  <a href="https://github.com/exoplanet-dev/exoplanet">
    <img src="https://img.shields.io/badge/GitHub-exoplanet--dev%2Fexoplanet-blue.svg?style=flat"></a>
  <a href="https://github.com/exoplanet-dev/exoplanet/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat"></a>
  <a href="https://zenodo.org/badge/latestdoi/138077978">
    <img src="https://zenodo.org/badge/138077978.svg"></a>
  <a
    href="https://docs.google.com/viewer?url=https://github.com/exoplanet-dev/paper/raw/master-pdf/paper/exoplanet.pdf">
    <img src="https://img.shields.io/badge/read-the_paper_draft-blue.svg?style=flat"></a>
  <a href="https://docs.exoplanet.codes/en/latest">
    <img src="https://img.shields.io/badge/read-the_docs-blue.svg?style=flat"></a>
  <br>
  <a href="https://travis-ci.com/exoplanet-dev/exoplanet">
    <img src="https://img.shields.io/travis/exoplanet-dev/exoplanet/master.svg?style=flat&label=tests"></a>
  <a href="https://dev.azure.com/foremanmackey/exoplanet/_build?definitionId=15">
    <img
      src="https://img.shields.io/azure-devops/build/foremanmackey/exoplanet/15/master?label=paper&style=flat"></a>
  <a href="https://dev.azure.com/foremanmackey/exoplanet/_build?definitionId=14">
    <img
      src="https://img.shields.io/azure-devops/build/foremanmackey/exoplanet/14/master?label=docs&style=flat"></a>
  <br>
  <a href="https://rodluger.github.io/starry">
    <img src="https://img.shields.io/badge/powered_by-starry-EB5368.svg?style=flat"></a>
  <a href="https://celerite.readthedocs.io">
    <img src="https://img.shields.io/badge/powered_by-celerite-EB5368.svg?style=flat"></a>
  <a href="https://docs.pymc.io">
    <img src="https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat"></a>
  <a href="http://www.astropy.org">
    <img src="https://img.shields.io/badge/powered_by-AstroPy-EB5368.svg?style=flat"></a>
  <a href="https://github.com/rodluger/corTeX">
    <img src="https://img.shields.io/badge/powered_by-corTeX-EB5368.svg?style=flat"></a>
</p>

*exoplanet* is a toolkit for probabilistic modeling of transit and/or
radial velocity observations of
[exoplanets](https://en.wikipedia.org/wiki/Exoplanet) and other
astronomical time series using [PyMC3](https://docs.pymc.io). *PyMC3* is
a flexible and high-performance model building language and inference
engine that scales well to problems with a large number of parameters.
*exoplanet* extends *PyMC3*'s language to support many of the custom
functions and distributions required when fitting exoplanet datasets.
These features include:

- A fast and robust solver for Kepler's equation.
- Scalable Gaussian Processes using
  [celerite](https://celerite.readthedocs.io).
- Fast and accurate limb darkened light curves using
  [starry](https://rodluger.github.io/starry).
- Common reparameterizations for [limb darkening
  parameters](https://arxiv.org/abs/1308.0009), and [planet radius and
  impact parameter](https://arxiv.org/abs/1811.04859).
- And many others!

All of these functions and distributions include methods for efficiently
calculating their *gradients* so that they can be used with
gradient-based inference methods like [Hamiltonian Monte
Carlo](https://arxiv.org/abs/1206.1901), [No U-Turns
Sampling](https://arxiv.org/abs/1111.4246), and [variational
inference](https://arxiv.org/abs/1603.00788). These methods tend to be
more robust than the methods more commonly used in astronomy (like
[ensemble samplers](https://emcee.readthedocs.io) and [nested
sampling](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/))
especially when the model has more than a few parameters. For many
exoplanet applications, *exoplanet* (the code) can improve the typical
performance by orders of magnitude.

*exoplanet* is being actively developed in [a public repository on
GitHub](https://github.com/exoplanet-dev/exoplanet) so if you have any trouble,
[open an issue](https://github.com/exoplanet-dev/exoplanet/issues) there.

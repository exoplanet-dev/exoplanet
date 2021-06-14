---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(reparameterization)=

# Reparameterization

+++

One thing that you'll often find when using `exoplanet` (and `PyMC3` or other libraries for gradient-based inference) is that performance of your inference can be quite sensitive to the parameterization of your problem.
For example, you might see warnings from `PyMC3` telling you to consider reparameterizing your model (because of divergences) or (worse!) you might find that sampling is unreasonably slow or the number of effective samples is very small.
These issues can almost always be solved by investigating any sources of non-linear degeneracies between parameters in your model or heavy tails in the posterior density.

[The Stan User's Guide](https://mc-stan.org/docs/2_27/stan-users-guide/reparameterization-section.html) includes some general advice about reparameterization that can be a useful place to start, but I've found that the best choices tend to be pretty problem specific.
Throughout these tutorials and the [Case Studies](https://gallery.exoplanet.codes) we have tried to select sensible parameterizations (sometimes after many experiments) and comment wherever we've chosen something non-standard.

Some of the trickiest parameters are things like orbital eccentricity, inclination, and other angles.
In the case of eccentricity, you can sometimes reparameterize in terms of an observable (such as transit duration, see the "Quick fits for TESS light curves" case study).
For angles it can sometimes be better to parameterize in terms of sums or differences of pairs of angles (see the example at {ref}`data-and-models/astrometry` for a demo).

As I learn more general advice for reparameterization of `exoplanet` models, I'll try to keep this page updated, but in the meantime, feel free to start [a "discussion" on the GitHub repository](https://github.com/exoplanet-dev/exoplanet/discussions) if you discover a particularly good or tricky parameterization problem.

```{code-cell}

```

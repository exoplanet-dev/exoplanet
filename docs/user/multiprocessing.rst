.. _multiprocessing:

Multiprocessing
===============

By default, when you sample with ``PyMC3``, it will try to run chains in
parallel using Python's `multiprocessing` module. This is pretty much always
what you want, but it can also cause some headaches (mostly on macOS or Windows)
for the large and computationally expensive models tackled by ``exoplanet``. In
particular, you might sometimes get hit by the "dreaded broken pipe" error where
your sampler crashes for no obvious reason or (worse!) you might find you
sampler hanging indefinitely before it even starts running. The official
``PyMC3`` solution is to use the ``mp_ctx="forkserver"`` option when calling
``pm.sample`` on macOS or Windows. Unfortunately, this (for reasons that I don't
totally understand) will often cause a huge performance hit that can increase
your runtime by orders of magnitude.

Throughout these documentation pages and for the Case Studies, we have tried to
design the example models such that you shouldn't run into issues with
multiprocessing but, if you do, please `open an issue on GitHub
<https://github.com/exoplanet-dev/exoplanet/issues>`_. For your own projects, if
you run into multiprocessing issues, you can try adjusting the ``mp_ctx`` and
``pickle_backend`` parameters (see the `PyMC3 docs
<https://docs.pymc.io/api/inference.html#pymc3.sampling.sample>`_), or as a last
resort, set ``cores=1`` to get serial sampling.

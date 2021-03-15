.. _theano:

Theano vs. Aesara
=================

*exoplanet* is built on top of the excellent `PyMC3 library
<https://docs.pymc.io/>`_ and it uses some of the more advanced features of
PyMC3 to enable the complex models required for exoplanet data analysis. As a
user of *exoplanet* you will become familiar with some of these details and, in
particular, you will find yourself interacting with PyMC3's backend directly.
When it was originally designed, PyMC3 was built on top of a library called
Theano, but the development of that library has since been taken over by the
PyMC developers and it was renamed to `Aesara <https://aesara.readthedocs.io>`_
(there are more details, and you can read about those in `this blog post from
the PyMC developers
<https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b>`_).
Theano and Aesara are not completely compatible so this transition has been a
little painful for downstream users, like you and the developers of *exoplanet*.
In the long run, we will switch over to using Aesara directly instead of Theano,
but in the meantime, since some users will have Theano installed and others will
have Aesara installed, we have developed a small library that can help smooth
this transition: `aesara-theano-fallback
<https://github.com/exoplanet-dev/aesara-theano-fallback>`_. This will be
installed automatically when you install *exoplanet*, but you can also install
it directly using:

.. code-block:: bash

    python -m pip install aesara-theano-fallback

Then, you can use this with syntax like the following:

.. code-block:: python

    import aesara_theano_fallback.tensor as tt

and:

.. code-block:: python

    from aesara_theano_fallback import aesara as theano

In both of these examples, the library will attempt to provide a consistent
interface regardless of which backend is installed. That being said, this has
only been designed to support the features used by *exoplanet* and `starry
<https://github.com/rodluger/starry>`_, so your milage might vary. Check out
`aesara-theano-fallback on GitHub for more information or to open issues
<https://github.com/exoplanet-dev/aesara-theano-fallback>`_.

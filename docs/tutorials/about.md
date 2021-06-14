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

# About these tutorials

+++

This and the following tutorials are automatically executed with every change of the code to make sure that they are always up to date with the code.
As a result, they are designed to require only a relatively small amount of computation time; when using `exoplanet` for research you will probably find that your runtimes are longer.
For more in-depth tutorials with real-world applications and real data, check out the [Case Studies page](https://gallery.exoplanet.codes).

At the top of each tutorial, you'll find a cell like the following that indicates the version of `exoplanet` that was used to generate the tutorial:

```{code-cell}
import exoplanet

exoplanet.utils.docs_setup()
print(f"exoplanet.__version__ = '{exoplanet.__version__}'")
```

That cell also includes a call to the `exoplanet.utils.docs_setup` function that will squash some warnings (these are generally caused by Theano/Aesara; see {ref}`theano` for more info) and set up our `matplotlib` style.

To exectute a tutorial on your own, you can click on the buttons at the top right or this page to launch the notebook using [Binder](https://mybinder.org) or download the `.ipynb` file directly.

```{code-cell}

```

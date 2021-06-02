---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Automatic differentation & gradient-based inference

```{code-cell}
import exoplanet

exoplanet.utils.docs_setup()
print(f"exoplanet.__version__ = '{exoplanet.__version__}'")
```

The major selling point of `exoplanet` compared to other similar libraries is that it integrates with the `PyMC3` probabilistic modeling framework.
`PyMC3` offers gradient-based inference algorithms (more on these below) that can be more computationally efficient (per effective sample) than other tools commonly used for probabilistic inference in astrophysics.
When I am writing this, gradient-based inference methodology is not widely used in astro because it can be difficult to compute the relevant gradients (derivatives of your log probability function with respect to the parameters).
`PyMC3` (and many other frameworks like it) handle this issue using [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation), a method (or collection of methods) for automatically propagating derivatives through your code.
It's beyond the scope of this tutorial to go into too many details about AD and most users of `exoplanet` shouldn't need to interact with this too much, but this should at least give you a little taste of the kinds of things AD can do for you and demonstrate how this translates into efficient inference with probabilistic models.

## Automatic differentation in Theano/Aesara

`PyMC3` is built on top of Theano/Aesara.

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from aesara_theano_fallback import aesara
import aesara_theano_fallback.tensor as at

x_ = at.vector("x")
y_ = at.exp(at.sin(2 * np.pi * x_ / 2.0))
func = aesara.function([x_], y_)

x = np.linspace(-3, 3, 500)
plt.plot(x, func(x))
plt.title("a silly function")
plt.ylabel("f(x)")
_ = plt.xlabel("x")
```

```{code-cell}
grad = aesara.function([x_], aesara.grad(at.sum(y_), x_))

plt.plot(x, func(x), label="f(x)")
plt.plot(x, grad(x), label="df(x)/dx")
plt.legend()
plt.title("a silly function and its derivative")
plt.ylabel("f(x); df(x)/dx")
_ = plt.xlabel("x")
```

```{code-cell}

```

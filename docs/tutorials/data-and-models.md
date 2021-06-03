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

(data-and-models)=

# Data & models

```{code-cell}
import exoplanet

exoplanet.utils.docs_setup()
print(f"exoplanet.__version__ = '{exoplanet.__version__}'")
```

This tutorial provides an overview of the kinds of datasets that `exoplanet` has been designed to model.
It's worth noting that this is certainly not a comprehensive list and `exoplanet` is designed to be *low-level* library that can be built on top of to support new use cases.

:::{tip}
For real world use cases and examples, check out the [Case Studies](https://gallery.exoplanet.codes) page.
:::

+++

## Overview

One primary interface to `exoplanet` is via the {class}`exoplanet.orbits.KeplerianOrbit` object (and its generalizations like {class}`exoplanet.orbits.TTVOrbit`).
The `KeplerianOrbit` object defines a set of independent (but possibly massive) bodies orbiting a central body on Keplerian trajectories.
These orbits are parameterized by orbital elements, and one of the primary uses of `exoplanet` is the measurement of these orbital elements given observations.

Given a `KeplerianOrbit`, users of `exoplanet` can evaluate things like the positions and velocities of all the bodies as a function of time.
For example, here's how you could define an orbit with a single body and plot the orbits:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo

# Define the orbit
orbit = xo.orbits.KeplerianOrbit(
    period=10.0,  # All times are in days
    t0=0.5,
    incl=0.5 * np.pi,  # All angles are in radians
    ecc=0.3,
    omega=-2.5,
    Omega=1.2,
    m_planet=0.05,  # All masses and distances are in Solar units
)

# Get the position and velocity as a function of time
t = np.linspace(0, 20, 5000)
x, y, z = orbit.get_relative_position(t)
vx, vy, vz = orbit.get_star_velocity(t)

# Plot the coordinates
# Note the use of `.eval()` throughout since the coordinates are all
# Aesara/Theano objects
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax = axes[0]
ax.plot(t, x.eval(), label="x")
ax.plot(t, y.eval(), label="y")
ax.plot(t, z.eval(), label="z")
ax.set_ylabel("position of orbiting body [$R_*$]")
ax.legend(fontsize=10, loc=1)

ax = axes[1]
ax.plot(t, vx.eval(), label="$v_x$")
ax.plot(t, vy.eval(), label="$v_y$")
ax.plot(t, vz.eval(), label="$v_z$")
ax.set_xlim(t.min(), t.max())
ax.set_xlabel("time [days]")
ax.set_ylabel("velocity of central [$R_*$/day]")
_ = ax.legend(fontsize=10, loc=1)
```

The key feture of `exoplanet` is that all of the parameters to a `KeplerianOrbit` can be `PyMC3` variables.
This means that these elements are now something that you can *infer*.
For example, if we want to fit for the orbital period, we can define a `PyMC3` model like the following:

```{code-cell}
import pymc3 as pm

with pm.Model():
    log_period = pm.Normal("log_period", mu=np.log(10), sigma=2.0)
    orbit = xo.orbits.KeplerianOrbit(
        period=pm.math.exp(log_period),  # ...
    )

    # Define the rest of you model using `orbit`...
```

In the following sections, we will go through some of the specific ways that you might use this `orbit` to define a model, but first some orbital conventions!

+++

## Orbital conventions

These orbits are specified specified with respect to a single central body (generally the most massive body) and then a system of non-interacting bodies orbiting the central.
This is a good parameterization for exoplanetary systems and binary stars, but it is sometimes not sufficient for systems with multiple massive bodies where the interactions will be important to the dynamics.

We follow a set of internally consistent orbital conventions that also attempts to respect as many of the established conventions of the stellar and exoplanetary fields as possible.
The location of a body on a Keplerian orbit with respect to the center of mass is given in the perifocal plane by

$$
\boldsymbol{r} =
  \left [
    \begin{array}{c}
      x\\
      y\\
      z\\
    \end{array}
  \right ].
$$

By construction, the orbit of the body in the perifocal frame is constrained entirely to the $x -y$ plane as shown by this figure:

![Orbital geometry](../_static/orbit3D.png)

The range of orbital convention choices mainly stems from how the location of the body in the perifocal frame is converted to the observer frame,

$$
\boldsymbol{R} =
  \left [
    \begin{array}{c}
      X\\
      Y\\
      Z\\
    \end{array}
  \right ].
$$

We choose to orient the $\hat{\boldsymbol{X}}$ unit vector in the north direction, $\hat{\boldsymbol{Y}}$ in the east direction, and $\hat{\boldsymbol{Z}}$ towards the observer. Under this convention, the inclination of an orbit is defined by the dot-product of the orbital angular momentum vector with $\hat{\boldsymbol{Z}}$, conforming to the common astrometric convention that a zero inclination ($i = 0$) orbit is face-on, with the test body orbiting in a counter-clockwise manner around the primary body.

In the stellar and exoplanetary fields, there is less agreement on which  segment of the line of nodes is defined as the *ascending* node.
We choose to define the ascending node as the point where the orbiting body crosses the plane of the sky moving *away* from the observer (i.e., crossing from $Z > 0$ to $Z< 0$).
This convention has historically been the choice of the visual binary field; the opposite convention occasionally appears in exoplanet and planetary studies.

To implement the transformation from perifocal frame to observer frame, we consider the rotation matrices

$$
  \boldsymbol{P}_x(\phi) = \left [
  \begin{array}{ccc}
    1 & 0 & 0 \\
    0 & \cos \phi & - \sin \phi \\
    0 & \sin \phi & \cos \phi \\
    \end{array}\right]
$$

$$
  \boldsymbol{P}_z (\phi) = \left [
  \begin{array}{ccc}
    \cos \phi & - \sin \phi & 0\\
    \sin \phi & \cos \phi & 0 \\
    0 & 0 & 1 \\
    \end{array}\right].
$$

which result in a *clockwise* rotation of the axes, as defined using the right hand rule.

This means when we look down the $z$-axis, for a positive angle $\phi$, it would be as if the $x$ and $y$ axes rotated clockwise.
In order to find out what defines counter-clockwise when considering the other rotations, we look to the right hand rule and cross products of the axes unit vectors.
Since $\hat{\boldsymbol{x}} \times \hat{\boldsymbol{y}} = \hat{\boldsymbol{z}}$, when looking down the $z$ axis the direction of the $x$-axis towards the $y$-axis defines counter clockwise.
Similarly, we have $\hat{\boldsymbol{y}} \times \hat{\boldsymbol{z}} = \hat{\boldsymbol{x}}$, and $\hat{\boldsymbol{z}} \times \hat{\boldsymbol{x}} = \hat{\boldsymbol{y}}$.

To convert a position $\boldsymbol{r}$ in the perifocal frame to the observer frame $\boldsymbol{R}$ (referencing the figure above), three rotations are required:
(i) a rotation about the $z$-axis through an angle $\omega$ so that the $x$-axis coincides with the line of nodes at the ascending node,
(ii) a rotation about the $x$-axis through an angle ($-i$) so that the two planes are coincident, and finally
(iii) a rotation about the $z$-axis through an angle $\Omega$. Applying these rotation matrices yields

$$
  \boldsymbol{R} =
  \boldsymbol{P}_z (\Omega) \boldsymbol{P}_x(-i) \boldsymbol{P}_z(\omega)
  \boldsymbol{r}.
$$

As a function of true anomaly $f$, the position in the observer frame is given by

$$
  \begin{array}{lc}
    X =& r [ \cos \Omega (\cos \omega \cos f - \sin \omega \sin f)  - \sin \Omega  \cos i (\sin \omega \cos f + \cos \omega \sin f) ] \\
    Y =& r [ \sin \Omega (\cos \omega \cos f - \sin \omega \sin f) + \cos \Omega \cos i(\sin \omega \cos f + \cos \omega \sin f) ] \\
    Z =& - r \sin i (\sin \omega \cos f + \cos \omega \sin f).\\
\end{array}
$$

Simplifying the equations using the sum angle identities, we find

$$
  \begin{array}{lc}
    X =& r (\cos \Omega \cos(\omega + f) - \sin(\Omega) \sin(\omega + f) \cos(i)) \\
    Y =& r (\sin \Omega \cos(\omega + f) + \cos(\Omega) \sin(\omega + f) \cos(i)) \\
    Z =& - r \sin(\omega + f) \sin(i).\\
\end{array}
$$

The central body is defined by any two of radius $R_\star$, mass $M_\star$, or density $\rho_\star$.
The third physical parameter can always be computed using the other two so `exoplanet` will throw an error if three are given (even if they are numerically self-consistent).

The orbits are then typically defined by (check out the docstring {class}`exoplanet.orbits.KeplerianOrbit` for all the options):

* the period $P$ or semi-major axis $a$ of the orbit,
* the time of conjunction $t_0$ or time of periastron passage $t_\mathrm{p}$,
* the planet's radius $R_\mathrm{pl}$ and mass $M_\mathrm{pl}$,
* the eccentricity $e$, argument of periastron $\omega$ and the longitude of ascending node $\Omega$, and
* the inclination of the orbital plane $i$ or the impact parameter $b$.

A `KeplerianOrbit` object in `exoplanet` will always be fully specified.
For example, if the orbit is defined using the period $P$, the semi-major axis $a$ will be computed using Kepler's third law.

+++

(data-and-models/rvs)=

## Radial velocities

A {class}`exoplanet.orbits.KeplerianOrbit` can be used to compute the expected radial velocity time series for a given set of parameters.
One typical parameterization for a radial velocity fit would look something like this:

```{code-cell}
import arviz as az
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt

# Create a dummy dataset
random = np.random.default_rng(1234)
t = np.sort(random.uniform(0, 50, 20))
rv_err = 0.01 + np.zeros_like(t)
rv_obs = 5.0 * np.sin(2 * np.pi * t / 10.0) + np.sqrt(
    0.05 ** 2 + rv_err ** 2
) * random.normal(size=len(t))

with pm.Model():

    # Period, semi-amplitude, and eccentricity
    log_period = pm.Normal("log_period", mu=np.log(10.0), sigma=1.0)
    period = pm.Deterministic("period", tt.exp(log_period))
    log_semiamp = pm.Normal("log_semiamp", mu=np.log(5.0), sd=2.0)
    semiamp = pm.Deterministic("semiamp", tt.exp(log_semiamp))
    ecc = pm.Uniform("ecc", lower=0, upper=1)

    # At low eccentricity, omega and the phase of periastron (phi) are
    # correlated so it can be best to fit in (omega ± phi) / 2
    plus = pmx.Angle("plus")
    minus = pmx.Angle("minus")
    phi = pm.Deterministic("phi", plus + minus)
    omega = pm.Deterministic("omega", plus - minus)

    # For non-zero eccentricity, it can sometimes be better to use
    # sqrt(e)*sin(omega) and sqrt(e)*cos(omega) as your parameters:
    #
    #     ecs = pmx.UnitDisk("ecs", testval=0.01 * np.ones(2))
    #     ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
    #     omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    #     phi = pmx.Angle("phi")

    # Jitter & the system mean velocity offset
    log_jitter = pm.Normal("log_jitter", mu=np.log(0.05), sd=5.0)
    zero_point = pm.Normal("zero_point", mu=0, sd=10.0)

    # Then we define the orbit
    tperi = pm.Deterministic("tperi", period * phi / (2 * np.pi))
    orbit = xo.orbits.KeplerianOrbit(
        period=period, t_periastron=tperi, ecc=ecc, omega=omega
    )

    # And define the RV model
    rv_model = zero_point + orbit.get_radial_velocity(t, K=semiamp)

    # Finally add in the observation model
    err = tt.sqrt(rv_err ** 2 + tt.exp(2 * log_jitter))
    pm.Normal("obs", mu=rv_model, sigma=rv_err, observed=rv_obs)

    soln = pmx.optimize(vars=[plus, minus, ecc])
    soln = pmx.optimize(soln)
    trace = pmx.sample(
        tune=1000,
        draws=1000,
        cores=2,
        chains=2,
        start=soln,
        return_inferencedata=True,
    )

az.summary(trace)
```

Instead of using `semiamp` as a parameter, we could have passed `m_planet` and `incl` as parameters to `KeplerianOrbit` and fit the physical orbit.
Similarly, if we we doing a joint fit with a transit light curve (see {ref}`data-and-models/combining` below) we could have parameterized in terms of `t0` (the time of a reference transit) instead of the phase of periastron.

As mentioned in the comments above and discussed in {ref}`reparameterization`, the performance of these fits can be problem specific and depend on the choice of parameters.
Therefore it is often worthwhile experimenting with different parameterizations, but this can be a good place to start for radial velocity-only fits.

+++

(data-and-models/astrometry)=

## Astrometry

Astrometric observations usually consist of measurements of the separation and position angle of the secondary star (or directly imaged exoplanet), relative to the primary star as a function of time, but `exoplanet` could also be used to model the motion of the center of light for and unresolved orbit.
The typical {class}`exoplanet.orbits.KeplerianOrbit` definition for and astrometric dataset will be similar to a radial velocity fit:

```{code-cell}
random = np.random.default_rng(5678)
t = np.sort(random.uniform(0.0, 22 * 365.25, 45))
rho_err = random.uniform(0.05, 0.1, len(t))
theta_err = random.uniform(0.05, 0.1, len(t))

with pm.Model():

    # Period, semi-major axis, eccentricity, and t0
    log_period = pm.Normal("log_period", mu=np.log(25.0 * 365.25), sigma=1.0)
    period = pm.Deterministic("period", tt.exp(log_period))
    log_a = pm.Normal("log_a", mu=np.log(0.3), sd=2.0)
    a = pm.Deterministic("a", tt.exp(log_a))
    ecc = pm.Uniform("ecc", lower=0, upper=1)
    tperi = pm.Normal("tperi", mu=3500.0, sigma=1000.0)

    # For astrometric orbits, a good choice of parameterization can be
    # (Omega ± omega) / 2
    plus = pmx.Angle("plus")
    minus = pmx.Angle("minus")
    Omega = pm.Deterministic("Omega", plus + minus)
    omega = pm.Deterministic("omega", plus - minus)

    # We'll use a uniform prior on cos(incl)
    cos_incl = pm.Uniform("cos_incl", lower=-1.0, upper=1.0, testval=0.3)
    incl = pm.Deterministic("incl", tt.arccos(cos_incl))

    # Then we define the orbit
    orbit = xo.orbits.KeplerianOrbit(
        a=a,
        t_periastron=tperi,
        period=period,
        incl=incl,
        ecc=ecc,
        omega=omega,
        Omega=Omega,
    )

    # And define the RV model
    rho_model, theta_model = orbit.get_relative_angles(t)

    # ================================================== #
    # Simulate data from the model for testing           #
    # You should remove the following lines in your code #
    # ================================================== #
    rho_obs, theta_obs = pmx.eval_in_model([rho_model, theta_model])
    rho_obs += rho_err * random.normal(size=len(t))
    theta_obs += theta_err * random.normal(size=len(t))
    # =============== end simulated data =============== #

    # Define the observation model this is simple for rho:
    pm.Normal("rho_obs", mu=rho_model, sd=rho_err, observed=rho_obs)

    # But we want to be cognizant of the fact that theta wraps so the following
    # is equivalent to
    #
    #   pm.Normal("obs_theta", mu=theta_model, observed=theta_obs, sd=theta_err)
    #
    # but takes into account the wrapping
    theta_diff = tt.arctan2(
        tt.sin(theta_model - theta_obs), tt.cos(theta_model - theta_obs)
    )
    pm.Normal("theta_obs", mu=theta_diff, sd=theta_err, observed=0.0)

    trace = pmx.sample(
        tune=1000,
        draws=1000,
        cores=2,
        chains=2,
        target_accept=0.95,
        return_inferencedata=True,
    )

az.summary(trace)
```

In this example, the units of `a` are arcseconds, but the {func}`exoplanet.orbits.KeplerianOrbit.get_relative_angles` function accepts a parallax argument if you have a constraint on the parallax of the system.
If you provide parallax as an argument, `a` should be provided in the usual units of Solar radii.

+++

(data-and-models/transits)=

## Transits, occultations, and eclipses

+++

(data-and-models/combining)=

## Combining datasets

```{code-cell}

```

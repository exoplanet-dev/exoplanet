
# coding: utf-8

import kplr
import numpy as np
import matplotlib.pyplot as plt
# from astropy.table import Table
from scipy.signal import medfilt

import tensorflow as tf
session = tf.InteractiveSession()

from exoplanet import transit, kepler, interp

import hemcee
import tfmodeling as tfm

# import celeriteflow as cf


# In[2]:

kepid = 11709124
client = kplr.API()
star = client.star(kepid)
t = []
y = []
yerr = []
quarter = []
texp = []
for i, lc in enumerate(star.get_light_curves(short_cadence=False)):
    with lc.open() as f:
        hdr = f[1].header
        texp0 = hdr["INT_TIME"] * hdr["NUM_FRM"] / (24. * 60. * 60.)
        data = f[1].data

    q = data["SAP_QUALITY"]
    t0 = data["TIME"]
    y0 = data["PDCSAP_FLUX"]
    m = (q == 0) & np.isfinite(t0) & np.isfinite(y0)
    mu = medfilt(y0[m], 101)
#     mu = np.median(y0[m])
    t.append(t0[m])
    y.append((y0[m] / mu - 1.0) * 1e3)
    yerr.append(data["PDCSAP_FLUX_ERR"][m] * 1e3 / mu)
    quarter.append(i + np.zeros(m.sum(), dtype=int))
    texp.append(texp0 + np.zeros(m.sum()))
t = np.ascontiguousarray(np.concatenate(t), dtype=np.float64)
y = np.ascontiguousarray(np.concatenate(y), dtype=np.float64)
yerr = np.ascontiguousarray(np.concatenate(yerr), dtype=np.float64)
quarter = np.ascontiguousarray(np.concatenate(quarter), dtype=int)
texp = np.median(np.concatenate(texp))


# In[3]:

kois = star.kois
spec = np.empty(len(kois),
                dtype=[("period", float), ("ror", float), ("t0", float), ("duration", float)])
for i, koi in enumerate(kois):
    spec["period"][i] = koi.koi_period
    spec["ror"][i] = koi.koi_ror
    spec["t0"][i] = koi.koi_time0bk % koi.koi_period
    spec["duration"][i] = koi.koi_duration / 24.0
spec = spec[np.argsort(spec["period"])]


# In[20]:

# gkf = Table.read("kepler_dr2_1arcsec.fits", format="fits").to_pandas()
# star = gkf[gkf.kepid.astype(int) == kepid].iloc[0]

# vals = dict()
# for k in ["radius", "mass"]:
#     mn = star[k] + star[k + "_err2"]
#     mx = star[k] + star[k + "_err1"]
#     log_mu = np.log(star[k])
#     log_err = 0.5 * ((log_mu - np.log(mn)) + (np.log(mx) - log_mu))
#     vals[k] = (log_mu, log_err)

# np.random.seed(42)
# N = 50000
# R = np.exp(vals["radius"][0] + vals["radius"][1] * np.random.randn(N))
# M = np.exp(vals["mass"][0] + vals["mass"][1] * np.random.randn(N))
# log_dens = np.log(0.75 * M / (np.pi * R**3))
# log_dens = (np.mean(log_dens), np.std(log_dens))
# print(log_dens)
log_dens = -1.4547524087769572, 0.3502230916468015


# In[21]:

t0 = spec["t0"][-1]
tau = spec["duration"][-1]
rng = (t0 - 1000 < t) & (t < t0 + 1000)
plt.plot(t[rng], y[rng], "k")


# In[6]:

spec


# In[7]:

T = tf.float64
np.random.seed(42)

G_grav = 2945.462538537765
factor = (G_grav / (3 * np.pi)) ** (1./3)

t_tensor = tf.constant(t[rng], dtype=T)
y_tensor = tf.constant(y[rng], dtype=T)
yerr_tensor = tf.constant(yerr[rng], dtype=T)

# Star
log_jitter = tfm.Parameter(
    2.0*np.log(np.median(yerr)),
    bounds=(2.0*np.log(0.01*np.median(yerr)), 2.0*np.log(10.0*np.median(yerr))),
    dtype=T)
flux = tfm.Parameter(0.0, bounds=(-2, 2), dtype=T)
q = tfm.Parameter(0.5 + np.zeros(2), bounds=(0, 1), dtype=T)
log_rho_star = tfm.Parameter(log_dens[0], dtype=T)
q1 = q.value[0]
q2 = q.value[1]
c1 = 2.0 * tf.sqrt(q1) * q2
c2 = tf.sqrt(q1) * (1.0 - 2.0 * q2)
ld = transit.QuadraticLimbDarkening(c1, c2)

# planets
pls = [2, 5]
n_pl = len(pls)
log_Ps = []
for p in spec["period"][pls]:
#     bounds = None
    bounds = (np.log(p - 1), np.log(p + 1))
    if p > 365:
        bounds = (np.log(10), np.log(100*365.))
    log_Ps.append(tfm.Parameter(np.log(p), bounds=bounds, dtype=T))
log_P = tf.stack([p.value for p in log_Ps])

log_r = tfm.Parameter(np.log(spec["ror"][pls]), dtype=T)  # bounds=(-10, 10), dtype=T)
t0 = tfm.Parameter(spec["t0"][pls], dtype=T)
b = tfm.Parameter(0.5 + np.zeros(n_pl), bounds=(-2.0, 2.0), dtype=T)
e = tfm.Parameter(np.random.uniform(0, 0.01, n_pl), bounds=(0, 1.0), dtype=T)
# e = tfm.Parameter(np.zeros(n_pl), dtype=T)
omega_vec = tfm.UnitVector(np.random.randn(n_pl, 2), dtype=T)

period = tf.expand_dims(tf.exp(log_P), 1)
ror = tf.exp(log_r.value)
a = factor * tf.exp((2.0 * log_P + log_rho_star.value) / 3.0)
cosi = tf.abs(b.value) / a
incl = tf.acos(cosi)

sin_omega = omega_vec.value[:, 0]
cos_omega = omega_vec.value[:, 1]
omega = tf.atan2(sin_omega, cos_omega)

# Set up the interpolation grid for each planet
nint = 7
dt = texp * np.linspace(-0.5, 0.5, 2*nint+1)[1:-1:2]
hp = 0.5 * period
grid = 1.5 * spec["duration"][pls][:, None] * np.linspace(-0.5, 0.5, int(np.ceil(5 * spec["duration"].max() / texp)))[None, :]
grid_phase = tf.constant(grid, dtype=T) + hp
grid = tf.constant(grid[:, :, None] + dt[None, None, :], dtype=T)

# Solve Kepler's equation on the grid
coords = a[None, :, None, None] * kepler.sky_position(
    period[:, :, None],
    tf.zeros((n_pl, 1, 1), dtype=T),  # <- t0
    e.value[:, None, None],
    omega[:, None, None],
    incl[:, None, None],
    grid)

# Compute the sky position and the transit light curve on the grid
d = tf.sqrt(tf.reduce_sum(tf.square(coords[:2]), axis=0))
delta = transit.transit_depth(ld, d, ror[:, None, None]+tf.zeros_like(d), direction=coords[2], n_integrate=100)
transit_grid = 1e3 * tf.reduce_mean(delta, axis=-1)

# Compute the time coordinate for each planet
arg = tf.add(t_tensor[None, :], (-t0.value[:, None] + hp))
phase = tf.mod(arg, period)   # - hp[:, None]
# p = period[:, None]  # -tf.ones(n_pl, dtype=T)
transit_model = flux.value - tf.reduce_sum(interp.interp1d(phase, grid_phase, transit_grid), axis=0)

# transit_model = flux.value - 1e3 * tf.reduce_mean(tf.reduce_sum(delta, axis=0), axis=1)

err2 = tf.square(yerr_tensor) + tf.exp(log_jitter.value)
chi2 = tf.reduce_sum(tf.square(transit_model - y_tensor) / err2 + tf.log(err2))
chi2 += tf.square((log_rho_star.value - log_dens[0]) / log_dens[1])

# Beta prior
alpha = 0.867
beta = 3.03
chi2 -= 2*tf.reduce_sum((alpha-1)*tf.log(e.value) + (beta-1)*tf.log(1-e.value))

log_prob = -0.5*chi2

params = [flux, log_jitter, q, log_rho_star] + log_Ps + [log_r, t0, b, e, omega_vec]
model = tfm.Model(log_prob, params, session=session)
grad = tf.gradients(log_prob, model.get_parameters())

opt = tf.contrib.opt.ScipyOptimizerInterface(chi2, model.get_parameters())

session.run(tf.global_variables_initializer())

print(session.run(grad))


# In[8]:

arg.shape, period.shape, phase.shape


# In[19]:

from tensorflow.python.client import timeline

session.run(grad)

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

session.run(grad, options=options, run_metadata=run_metadata)
session.run(grad, options=options, run_metadata=run_metadata)

fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline_01.json', 'w') as f:
    f.write(chrome_trace)

get_ipython().magic('timeit session.run(grad)')

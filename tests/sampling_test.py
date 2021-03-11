# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm
import pytest

from exoplanet.sampling import sample


@pytest.mark.filterwarnings("ignore:The number of samples")
def test_full_adapt_sampling(seed=289586):
    np.random.seed(seed)

    L = np.random.randn(5, 5)
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    L[np.triu_indices_from(L, 1)] = 0.0

    with pm.Model():
        pm.MvNormal("a", mu=np.zeros(len(L)), chol=L, shape=len(L))
        sample(draws=10, tune=1000, random_seed=seed, cores=1, chains=1)

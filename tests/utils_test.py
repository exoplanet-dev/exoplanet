import numpy as np
import pymc3 as pm

from exoplanet.utils import eval_in_model


def test_eval_in_model(seed=123409):
    np.random.seed(seed)
    x_val = np.random.randn(5, 3)
    x_val2 = np.random.randn(5, 3)
    with pm.Model():
        x = pm.Normal("x", shape=x_val.shape, testval=x_val)
        assert np.allclose(eval_in_model(x), x_val)
        assert np.allclose(eval_in_model(x, {"x": x_val2}), x_val2)

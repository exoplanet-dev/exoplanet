import numpy as np
import pymc3 as pm
import pytest
import theano.tensor as tt

from exoplanet import optim as op
from exoplanet.optim import optimize

try:
    import torch
except ImportError:
    torch = None


def test_optimize(seed=1234):
    np.random.seed(seed)
    x_val = np.random.randn(5, 3)
    with pm.Model():
        pm.Normal("x", shape=x_val.shape, testval=x_val)
        soln1 = optimize(verbose=False)
        soln2, info = optimize(soln1, return_info=True, verbose=False)

    assert np.allclose(soln1["x"], 0.0)
    assert np.allclose(soln2["x"], 0.0)
    assert info.success


def test_optimize_exception(capsys):
    with pm.Model():
        cov = pm.Normal("cov", mu=np.eye(5), shape=(5, 5))
        chol = tt.slinalg.Cholesky(on_error="raise")(cov)
        pm.MvNormal("x", mu=np.zeros(5), chol=chol, shape=5)
        with pytest.raises(np.linalg.LinAlgError):
            optimize({"cov": np.zeros((5, 5))}, verbose=False)
        captured = capsys.readouterr()
        assert "array:" in captured.out
        assert "point:" in captured.out


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lr": 1e-4},
        {"lr": 1e-4, "betas": [0.92, 0.96]},
        {"lr": 1e-4, "betas": [0.92, 0.96], "eps": 1e-3},
        {"lr": 1e-4, "weight_decay": 0.1},
        {"amsgrad": True},
    ],
)
def test_adam(kwargs, seed=20200520):
    np.random.seed(seed)
    x0 = np.random.randn(2)

    x_torch = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([x_torch], **kwargs)

    with pm.Model():
        x = pm.Flat("x", shape=2, testval=x0)
        pm.Potential("rosenbrock", -rosenbrock(x))
        for obj, point in op.optimize_iterator(
            op.Adam(**kwargs), 100, vars=[x]
        ):
            optimizer.zero_grad()
            loss = rosenbrock(x_torch)
            loss.backward()
            optimizer.step()
            assert np.allclose(x_torch.detach().numpy(), point["x"])

# ref: https://github.com/pymc-devs/pymc3/blob/master/pymc3/tests/conftest.py

import pytest
from exoplanet.compat import change_flags


@pytest.fixture(scope="package", autouse=True)
def theano_config():
    flags = dict(compute_test_value="off")
    config = change_flags(**flags)
    with config:
        yield

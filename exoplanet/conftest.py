# -*- coding: utf-8 -*-
# ref: https://github.com/pymc-devs/pymc3/blob/master/pymc3/tests/conftest.py

import theano
import pytest


@pytest.fixture(scope="package", autouse=True)
def theano_config():
    config = theano.configparser.change_flags(compute_test_value='off')
    with config:
        yield

# -*- coding: utf-8 -*-

import numpy as np
import pytest

from .base_test import _Base
from .deprecated import RadiusImpact, get_joint_radius_impact


class TestPhyscial(_Base):
    random_seed = 20190919

    def test_radius_impact(self):
        min_radius = 0.01
        max_radius = 0.1
        with self._model():
            with pytest.warns(DeprecationWarning):
                dist = RadiusImpact(
                    "rb", min_radius=min_radius, max_radius=max_radius
                )

            # Test random sampling
            samples = dist.random(size=100)
            assert np.shape(samples) == (100, 2)
            assert np.all(
                (min_radius <= samples[:, 0]) & (samples[:, 0] <= max_radius)
            )

            logp = (
                RadiusImpact.dist(min_radius=min_radius, max_radius=max_radius)
                .logp(samples)
                .eval()
                .flatten()
            )
            assert np.all(np.isfinite(logp))
            assert np.allclose(logp[0], logp)

            trace = self._sample()

        r = trace["rb"][:, 0]
        b = trace["rb"][:, 1]

        # Make sure that the physical constraints are satisfied
        assert np.all((r <= max_radius) & (min_radius <= r))
        assert np.all((b >= 0) & (b <= 1 + r))

    def test_get_joint_radius_impact(self):
        min_radius = 0.01
        max_radius = 0.1
        with self._model() as model:
            with pytest.warns(DeprecationWarning):
                r, b = get_joint_radius_impact(
                    min_radius=min_radius, max_radius=max_radius
                )
            assert model.r is r
            assert model.b is b
            trace = self._sample()

        r = trace["r"]
        b = trace["b"]

        # Make sure that the physical constraints are satisfied
        assert np.all((r <= max_radius) & (min_radius <= r))
        assert np.all((b >= 0) & (b <= 1 + r))

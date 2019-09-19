# -*- coding: utf-8 -*-

__all__ = ["TTVOrbit", "compute_expected_transit_times"]

import numpy as np
import theano.tensor as tt

from .keplerian import KeplerianOrbit


def compute_expected_transit_times(min_time, max_time, period, t0):
    """Compute the expected transit times within a dataset

    Args:
        min_time (float): The start time of the dataset
        max_time (float): The end time of the dataset
        period (array): The periods of the planets
        t0 (array): The reference transit times for the planets

    Returns:
        A list of arrays of expected transit times for each planet

    """
    periods = np.atleast_1d(period)
    t0s = np.atleast_1d(t0)
    transit_times = []
    for period, t0 in zip(periods, t0s):
        min_ind = np.floor((min_time - t0) / period)
        max_ind = np.ceil((max_time - t0) / period)
        times = t0 + period * np.arange(min_ind, max_ind, 1)
        times = times[(min_time <= times) & (times <= max_time)]
        transit_times.append(times)
    return transit_times


class TTVOrbit(KeplerianOrbit):
    """A generalization of a Keplerian orbit with transit timing variations

    Only one of the arguments ``ttvs`` or ``transit_times`` can be given and
    the other will be computed from the one that was provided.

    In practice the way this works is that the time axis is shifted to account
    for the TTVs before solving for a standard Keplerian orbit. To identify
    which times corrorspond to which transits, this will find the closest
    labelled transit to the timestamp and adjust the timestamp accordingly.
    This means that things will go *very badly* if the time between
    neighboring transits is larger than ``2*period``.

    Args:
        ttvs: A list (with on entry for each planet) of "O-C" vectors for each
            transit of each planet in units of days. "O-C" means the
            difference between the observed transit time and the transit time
            expected for a regular periodic orbit.
        transit_times: A list (with on entry for each planet) of transit times
            for each transit of each planet in units of days. These times will
            be used to compute the implied (least squares) ``ttv_period`` and
            ``t0``. It is possible to supply a separate ``period`` parameter
            that will set the shape of the transits, but care should be taken
            to make sure that ``period`` and ``ttv_period`` don't diverge
            because things will break if the time between neighboring transits
            is larger than ``2*period``.

    """

    def __init__(self, *args, **kwargs):
        ttvs = kwargs.pop("ttvs", None)
        transit_times = kwargs.pop("transit_times", None)
        if ttvs is None and transit_times is None:
            raise ValueError(
                "one of 'ttvs' or 'transit_times' must be " "defined"
            )
        if ttvs is not None:
            self.ttvs = [tt.as_tensor_variable(ttv) for ttv in ttvs]

        else:
            # If transit times are given, compute the least squares period and
            # t0 based on these times.
            self.transit_times = []
            self.ttvs = []
            period = []
            t0 = []
            for i, times in enumerate(transit_times):
                times = tt.as_tensor_variable(times)

                N = times.shape[0]
                AT = tt.stack(
                    (tt.arange(N, dtype=times.dtype), tt.ones_like(times)),
                    axis=0,
                )
                A = tt.transpose(AT)
                ATA = tt.dot(AT, A)
                ATy = tt.dot(AT, times)
                w = tt.slinalg.solve_symmetric(ATA, ATy)
                expect = tt.dot(w, AT)

                period.append(w[0])
                t0.append(w[1])
                self.ttvs.append(times - expect)
                self.transit_times.append(times)

            kwargs["t0"] = tt.stack(t0)

            # We'll have two different periods: one that is the mean difference
            # between transit times and one that is a parameter that sets the
            # transit shape. If a "period" parameter is not given, these will
            # be the same. Users will probably want to put a prior relating the
            # two periods if they use separate values.
            self.ttv_period = tt.stack(period)
            given_period = kwargs.pop("period", None)
            if given_period is None:
                kwargs["period"] = self.ttv_period

        super(TTVOrbit, self).__init__(*args, **kwargs)
        self._base_time = 0.5 - self.t0 / self.period

        if ttvs is not None:
            self.ttv_period = self.period
            self.transit_times = [
                self.t0[i] + self.period[i] * tt.arange(ttv.shape[0]) + ttv
                for i, ttv in enumerate(self.ttvs)
            ]

        # Set up a histogram for identifying the transit offsets
        self._bin_edges = [
            tt.concatenate(
                (
                    [tts[0] - 0.5 * self.ttv_period[i]],
                    0.5 * (tts[1:] + tts[:-1]),
                    [tts[-1] + 0.5 * self.ttv_period[i]],
                )
            )
            for i, tts in enumerate(self.transit_times)
        ]
        self._bin_values = [
            tt.concatenate(([0], self.ttvs[i], [0]))
            for i in range(len(self.ttvs))
        ]

    def _get_model_dt(self, t):
        vals = []
        for i in range(len(self.ttvs)):
            inds = tt.extra_ops.searchsorted(self._bin_edges[i], t)
            vals.append(self._bin_values[i][inds])
        return tt.stack(vals, -1)

    def _warp_times(self, t):
        # This is the key function that takes the TTVs into account by
        # stretching the time axis around each transit
        return tt.shape_padright(t) - self._get_model_dt(t)

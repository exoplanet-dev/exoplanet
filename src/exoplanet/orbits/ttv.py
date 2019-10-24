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
        transit_inds: A list of integer value tensors giving the transit
            number for each transit in ``transit_times'' or ``ttvs``. This is
            useful when not all transits are observed. This should always be
            zero indexed.
        delta_log_period: If using the ``transit_times`` argument, this
            parameter specifies the difference (in natural log) between the
            leqast squares period and the effective period of the transit.

    """

    def __init__(self, *args, **kwargs):
        ttvs = kwargs.pop("ttvs", None)
        transit_times = kwargs.pop("transit_times", None)
        transit_inds = kwargs.pop("transit_inds", None)
        if ttvs is None and transit_times is None:
            raise ValueError(
                "one of 'ttvs' or 'transit_times' must be " "defined"
            )
        if ttvs is not None:
            self.ttvs = [tt.as_tensor_variable(ttv, ndim=1) for ttv in ttvs]
            if transit_inds is None:
                self.transit_inds = [
                    tt.arange(ttv.shape[0]) for ttv in self.ttvs
                ]
            else:
                self.transit_inds = [
                    tt.cast(tt.as_tensor_variable(inds, ndim=1), "int64")
                    for inds in transit_inds
                ]

        else:
            # If transit times are given, compute the least squares period and
            # t0 based on these times.
            self.transit_times = []
            self.ttvs = []
            self.transit_inds = []
            period = []
            t0 = []
            for i, times in enumerate(transit_times):
                times = tt.as_tensor_variable(times, ndim=1)
                if transit_inds is None:
                    inds = tt.arange(times.shape[0])
                else:
                    inds = tt.cast(
                        tt.as_tensor_variable(transit_inds[i]), "int64"
                    )
                self.transit_inds.append(inds)

                # A convoluted version of linear regression; don't ask
                N = times.shape[0]
                sumx = tt.sum(inds)
                sumx2 = tt.sum(inds ** 2)
                sumy = tt.sum(times)
                sumxy = tt.sum(inds * times)
                denom = N * sumx2 - sumx ** 2
                slope = (N * sumxy - sumx * sumy) / denom
                intercept = (sumx2 * sumy - sumx * sumxy) / denom
                expect = intercept + inds * slope

                period.append(slope)
                t0.append(intercept)
                self.ttvs.append(times - expect)
                self.transit_times.append(times)

            kwargs["t0"] = tt.stack(t0)

            # We'll have two different periods: one that is the mean difference
            # between transit times and one that is a parameter that sets the
            # transit shape. If a "period" parameter is not given, these will
            # be the same. Users will probably want to put a prior relating the
            # two periods if they use separate values.
            self.ttv_period = tt.stack(period)
            if "period" not in kwargs:
                if "delta_log_period" in kwargs:
                    kwargs["period"] = tt.exp(
                        tt.log(self.ttv_period)
                        + kwargs.pop("delta_log_period")
                    )
                else:
                    kwargs["period"] = self.ttv_period

        super(TTVOrbit, self).__init__(*args, **kwargs)

        if ttvs is not None:
            self.ttv_period = self.period
            self.transit_times = [
                self.t0[i] + self.period[i] * self.transit_inds[i] + ttv
                for i, ttv in enumerate(self.ttvs)
            ]

        # Compute the full set of transit times
        self.all_transit_times = []
        for i, inds in enumerate(self.transit_inds):
            expect = self.t0[i] + self.period[i] * tt.arange(inds.max() + 1)
            self.all_transit_times.append(
                tt.set_subtensor(expect[inds], self.transit_times[i])
            )

        # Set up a histogram for identifying the transit offsets
        self._bin_edges = [
            tt.concatenate(
                (
                    [tts[0] - 0.5 * self.ttv_period[i]],
                    0.5 * (tts[1:] + tts[:-1]),
                    [tts[-1] + 0.5 * self.ttv_period[i]],
                )
            )
            for i, tts in enumerate(self.all_transit_times)
        ]
        self._bin_values = [
            tt.concatenate(([tts[0]], tts, [tts[-1]]))
            for i, tts in enumerate(self.all_transit_times)
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

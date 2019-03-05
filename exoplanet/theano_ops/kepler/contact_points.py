# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CircularContactPointsOp", "ContactPointsOp"]


import numpy as np

import theano
from theano import gof
import theano.tensor as tt

from .find_roots import CircularRootFinder, GeneralRootFinder


class CircularContactPointsOp(tt.Op):

    __props__ = ("tol", )
    num_inputs = 4

    def __init__(self, tol=1e-6, **kwargs):
        self.tol = float(tol)
        super(CircularContactPointsOp, self).__init__(**kwargs)

    def make_node(self, *args):
        if len(args) != self.num_inputs:
            raise ValueError("expected {0} inputs".format(self.num_input))
        dtype = theano.config.floatX
        in_args = []
        for a in args:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        ndim = in_args[0].ndim
        out_args = [
            tt.TensorType(dtype=dtype,
                          broadcastable=[False] * ndim)()
            for i in range(4)]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0], shapes[0]

    def perform(self, node, inputs, outputs):
        a, i, r, R = inputs
        n_pl = a.size

        results = []
        for n in range(n_pl):
            solver = CircularRootFinder(a.flat[n], i.flat[n], tol=self.tol)
            roots = np.concatenate([solver.get_contact_points(L)
                                    for L in [R.flat[n] - r.flat[n],
                                              R.flat[n] + r.flat[n]]])
            results.append(np.sort(np.array(roots).flatten()))

        for m, roots in enumerate(zip(*results)):
            outputs[m][0] = np.array(roots).reshape(a.shape)


class ContactPointsOp(CircularContactPointsOp):
    num_inputs = 6

    def perform(self, node, inputs, outputs):
        a, e, w, i, r, R = inputs
        n_pl = a.size

        results = []
        for n in range(n_pl):
            solver = GeneralRootFinder(a.flat[n], e.flat[n], w.flat[n],
                                       i.flat[n], tol=self.tol)
            roots = np.concatenate([solver.get_contact_points(L)
                                    for L in [R.flat[n] - r.flat[n],
                                              R.flat[n] + r.flat[n]]])
            results.append(np.sort(np.array(roots).flatten()))

        for m, roots in enumerate(zip(*results)):
            outputs[m][0] = np.array(roots).reshape(a.shape)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CircularContactPointsOp", "ContactPointsOp"]

import pkg_resources

import theano
from theano import gof
import theano.tensor as tt


class CircularContactPointsOp(gof.COp):
    params_type = gof.ParamsType(
        maxiter=theano.scalar.int64,
        tol=theano.scalar.float64,
    )
    __props__ = ("tol", "maxiter")
    num_inputs = 4
    func_file = "./circular_contact_points.cc"
    func_name = "APPLY_SPECIFIC(circular_contact_points)"

    def __init__(self, tol=1e-12, maxiter=1000, **kwargs):
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        super(CircularContactPointsOp, self).__init__(
            self.func_file, self.func_name)

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def c_headers(self, compiler):
        return ["theano_helpers.h", "contact_points.h"]

    def c_header_dirs(self, compiler):
        return [
            pkg_resources.resource_filename(__name__, "include"),
        ]

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
        out_args = [in_args[0].type() for i in range(4)]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0], shapes[0]


class ContactPointsOp(CircularContactPointsOp):
    num_inputs = 6
    func_file = "./contact_points.cc"
    func_name = "APPLY_SPECIFIC(contact_points)"

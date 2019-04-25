# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ContactPointsOp"]

import pkg_resources

import theano
from theano import gof
import theano.tensor as tt

from ..build_utils import get_cache_version, get_compile_args


class ContactPointsOp(gof.COp):
    num_inputs = 7
    params_type = gof.ParamsType(
        tol=theano.scalar.float64,
    )
    __props__ = ("tol", )
    func_file = "./contact.cc"
    func_name = "APPLY_SPECIFIC(contact)"

    def __init__(self, tol=1e-10, **kwargs):
        self.tol = float(tol)
        super(ContactPointsOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_compile_args(self, compiler):
        return get_compile_args(compiler)

    def c_headers(self, compiler):
        return ["theano_helpers.h", "contact_points.h"]

    def c_header_dirs(self, compiler):
        return [pkg_resources.resource_filename(__name__, "include")]

    def make_node(self, *args):
        if len(args) != self.num_inputs:
            raise ValueError("expected {0} inputs".format(self.num_inputs))
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
                          broadcastable=[False] * ndim)(),
            tt.TensorType(dtype=dtype,
                          broadcastable=[False] * ndim)(),
            tt.TensorType(dtype="int32",
                          broadcastable=[False] * ndim)()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0]

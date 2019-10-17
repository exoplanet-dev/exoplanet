# -*- coding: utf-8 -*-

__all__ = ["ContactPointsOp"]

import theano
import theano.tensor as tt
from theano import gof

from ..build_utils import get_cache_version, get_compile_args, get_header_dirs


class ContactPointsOp(gof.COp):
    num_inputs = 7
    params_type = gof.ParamsType(tol=theano.scalar.float64)
    __props__ = ("tol",)
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
        return ["exoplanet/theano_helpers.h", "exoplanet/contact_points.h"]

    def c_header_dirs(self, compiler):
        return get_header_dirs(eigen=False)

    def make_node(self, *args):
        if len(args) != self.num_inputs:
            raise ValueError("expected {0} inputs".format(self.num_inputs))
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [
            in_args[0].type(),
            in_args[0].type(),
            tt.zeros_like(in_args[0], dtype="int32").type(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[0], shapes[0], shapes[0]

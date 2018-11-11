# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryBaseOp"]

import sys

import pkg_resources

import theano
from theano import gof
import theano.tensor as tt


class StarryBaseOp(gof.COp):

    __props__ = ()
    num_input = 0
    output_ndim = ()
    func_file = None
    func_name = None

    def __init__(self):
        super(StarryBaseOp, self).__init__(self.func_file, self.func_name)

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def c_headers(self, compiler):
        return ["theano_helpers.h"]

    def c_header_dirs(self, compiler):
        return [
            pkg_resources.resource_filename(__name__, "include"),
            pkg_resources.resource_filename(__name__, "starry/starry"),
            pkg_resources.resource_filename(__name__, "starry/lib/eigen_3.3.3")
        ]

    def c_compile_args(self, compiler):
        opts = ["-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

        if not any(f.startswith("-std=") for f in opts):
            for flag in ["-std=c++14", "-std=c++11"]:
                if compiler.has_flag(flag):
                    opts.append(flag)
                    break
            else:
                raise RuntimeError("c++11 or c++14 support is required")

        return opts

    def make_node(self, *args):
        if len(args) != self.num_input:
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
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False] * ndim)()
            for ndim in self.output_ndim]
        return gof.Apply(self, in_args, out_args)

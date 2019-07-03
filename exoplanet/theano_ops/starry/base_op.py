# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryBaseOp"]

import pkg_resources

from theano import gof

from ..build_utils import get_compile_args, get_cache_version


class StarryBaseOp(gof.COp):

    __props__ = ()
    func_file = None
    func_name = None

    def __init__(self):
        super(StarryBaseOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_headers(self, compiler):
        return ["theano_helpers.h", "integrate.h", "functors.h",
                "ellip.h", "limbdark.h", "utils.h"]

    def c_header_dirs(self, compiler):
        return [
            pkg_resources.resource_filename(__name__, "include"),
            pkg_resources.resource_filename(__name__, "include/eigen_3.3.3")
        ]

    def c_compile_args(self, compiler):
        return get_compile_args(compiler)
